use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{ConnectInfo, Path, State},
    http::{HeaderMap, StatusCode},
    middleware::{from_fn, from_fn_with_state},
    response::sse::{Event, KeepAlive},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
};

use athenas_core::ServerConfig;
use athenas_inference::{
    Backend, BackendFactory, ChatMessage, ChatRequest, CompletionRequest, ContentPart,
    EmbeddingRequest, MessageContent, ModelLoadConfig, Role, StreamChunk, TokenizeRequest,
};

use crate::api_keys::{AuthResult, SharedApiKeyManager};
use crate::audit_log::{AuditEntry, SharedAuditLogger};
use crate::metrics::{metrics_middleware, SharedMetrics};
use crate::middleware::{
    ip_filter_middleware, rate_limit_middleware, request_logging_middleware, IpFilterConfig,
    SharedRateLimiter,
};
use crate::model_manager::SharedModelManager;
use crate::model_router::SharedModelRouter;
use crate::semantic_cache::SharedSemanticCache;
use crate::session_manager::{SessionInfo, SharedSessionManager};
use crate::slot_manager::SlotManager;
use crate::vector_store::SharedVectorStore;

#[derive(Clone)]
struct AppState {
    model_manager: SharedModelManager,
    session_manager: SharedSessionManager,
    slot_manager: Option<Arc<SlotManager>>,
    api_key_manager: Option<SharedApiKeyManager>,
    model_router: Option<SharedModelRouter>,
    audit_logger: Option<SharedAuditLogger>,
    vector_store: Option<SharedVectorStore>,
    semantic_cache: Option<SharedSemanticCache>,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    start_time: std::time::Instant,
    /// Number of requests currently waiting for a semaphore permit.
    waiting_requests: Arc<std::sync::atomic::AtomicU32>,
    /// Number of requests currently holding a permit (active inference).
    active_requests: Arc<std::sync::atomic::AtomicU32>,
    /// Whether to include queue position headers in responses.
    queue_visibility: bool,
}

/// RAII guard that decrements the active request counter when dropped.
/// Wraps an `OwnedSemaphorePermit` so the semaphore is also released.
struct PermitGuard {
    permit: tokio::sync::OwnedSemaphorePermit,
    active_requests: Arc<std::sync::atomic::AtomicU32>,
}

impl Drop for PermitGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl std::ops::Deref for PermitGuard {
    type Target = tokio::sync::OwnedSemaphorePermit;
    fn deref(&self) -> &Self::Target {
        &self.permit
    }
}

/// Acquire a semaphore permit with queue tracking.
/// Returns `Some((guard, queue_position))` on success, or `None` if the
/// semaphore is closed. When `queue_visibility` is enabled, the caller
/// should include `X-Queue-Position` and `X-Active-Requests` headers in
/// the response.
async fn acquire_with_tracking(state: &AppState) -> Option<(PermitGuard, u32)> {
    use std::sync::atomic::Ordering;

    // Try a non-blocking acquire first (fast path — no queue)
    if let Ok(permit) = state.semaphore.clone().try_acquire_owned() {
        state.active_requests.fetch_add(1, Ordering::Relaxed);
        let guard = PermitGuard {
            permit,
            active_requests: state.active_requests.clone(),
        };
        return Some((guard, 0));
    }

    // Slow path — we're going to wait. Track our position.
    let queue_position = state.waiting_requests.fetch_add(1, Ordering::Relaxed) + 1;

    let permit_result = state.semaphore.clone().acquire_owned().await;

    state.waiting_requests.fetch_sub(1, Ordering::Relaxed);
    state.active_requests.fetch_add(1, Ordering::Relaxed);

    match permit_result {
        Ok(permit) => {
            let guard = PermitGuard {
                permit,
                active_requests: state.active_requests.clone(),
            };
            Some((guard, queue_position))
        }
        Err(_) => None,
    }
}

/// Build response headers with queue information if visibility is enabled.
fn queue_headers(state: &AppState, queue_position: u32) -> Vec<(String, String)> {
    use std::sync::atomic::Ordering;

    if !state.queue_visibility {
        return Vec::new();
    }

    let active = state.active_requests.load(Ordering::Relaxed);
    let waiting = state.waiting_requests.load(Ordering::Relaxed);

    vec![
        ("X-Queue-Position".to_string(), queue_position.to_string()),
        ("X-Active-Requests".to_string(), active.to_string()),
        ("X-Waiting-Requests".to_string(), waiting.to_string()),
    ]
}

/// Attach queue visibility headers to a response.
fn with_queue_headers(state: &AppState, queue_position: u32, mut resp: Response) -> Response {
    let headers = queue_headers(state, queue_position);
    if headers.is_empty() {
        return resp;
    }
    let h = resp.headers_mut();
    for (k, v) in headers {
        if let Ok(name) = axum::http::HeaderName::from_bytes(k.as_bytes()) {
            if let Ok(val) = axum::http::HeaderValue::from_str(&v) {
                h.insert(name, val);
            }
        }
    }
    resp
}

#[allow(clippy::too_many_arguments)]
pub fn create_router(
    model_manager: SharedModelManager,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    rate_limiter: SharedRateLimiter,
    session_manager: SharedSessionManager,
    slot_manager: Option<Arc<SlotManager>>,
    api_key_manager: Option<SharedApiKeyManager>,
    model_router: Option<SharedModelRouter>,
    audit_logger: Option<SharedAuditLogger>,
    vector_store: Option<SharedVectorStore>,
    semantic_cache: Option<SharedSemanticCache>,
    config: &ServerConfig,
) -> Router {
    let state = AppState {
        model_manager,
        session_manager,
        slot_manager,
        api_key_manager,
        model_router,
        audit_logger,
        vector_store,
        semantic_cache,
        metrics: metrics.clone(),
        semaphore,
        start_time: std::time::Instant::now(),
        waiting_requests: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        active_requests: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        queue_visibility: config.queue_visibility,
    };

    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/responses", post(responses))
        .route("/v1/queue", get(queue_status))
        .route("/v1/models", get(list_models))
        .route("/v1/models/load", post(load_model_endpoint))
        .route("/v1/models/unload", post(unload_model_endpoint))
        .route("/v1/models/default", post(set_default_model_endpoint))
        // Session management endpoints
        .route("/v1/sessions", post(create_session).get(list_sessions))
        .route("/v1/sessions/:id", get(get_session).delete(delete_session))
        .route("/v1/sessions/:id/messages", get(get_session_messages))
        .route("/v1/sessions/:id/system", post(set_session_system_prompt))
        .route("/v1/sessions/purge", post(purge_expired_sessions))
        // Slot management endpoints
        .route("/v1/slots", get(list_slots))
        .route("/v1/slots/:id/save", post(save_slot))
        .route("/v1/slots/:id/restore", post(restore_slot))
        .route("/v1/slots/:id/erase", post(erase_slot))
        .route("/v1/files", post(upload_file))
        .route("/v1/audio/transcriptions", post(transcribe_audio))
        // API Key management endpoints
        .route("/v1/keys", post(create_api_key).get(list_api_keys))
        .route("/v1/keys/:id", get(get_api_key).delete(delete_api_key))
        .route("/v1/keys/:id/revoke", post(revoke_api_key))
        .route("/v1/keys/:id/usage", get(get_api_key_usage))
        // Model routing endpoints
        .route("/v1/routing/aliases", post(create_alias).get(list_aliases))
        .route("/v1/routing/chains", post(create_chain).get(list_chains))
        .route("/v1/routing/health", get(routing_health))
        // Audit log endpoints
        .route("/v1/audit/logs", get(query_audit_logs))
        .route("/v1/audit/stats", get(audit_stats))
        // Vector store endpoints
        .route("/v1/vector/add", post(vs_add_document))
        .route("/v1/vector/add-batch", post(vs_add_batch))
        .route("/v1/vector/search", post(vs_search))
        .route("/v1/vector/documents", get(vs_list_documents))
        .route(
            "/v1/vector/documents/:id",
            get(vs_get_document).delete(vs_delete_document),
        )
        .route("/v1/vector/clear", post(vs_clear))
        .route("/v1/vector/stats", get(vs_stats))
        // Semantic cache endpoints
        .route("/v1/cache/stats", get(cache_stats))
        .route("/v1/cache/clear", post(cache_clear))
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

    // IP filter (only if allowlist or denylist is configured)
    let has_ip_filter = !config.ip_allowlist.is_empty() || !config.ip_denylist.is_empty();
    if has_ip_filter {
        let ip_filter =
            IpFilterConfig::new(config.ip_allowlist.clone(), config.ip_denylist.clone());
        router = router.layer(from_fn_with_state(ip_filter, ip_filter_middleware));
    }

    router = router
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(config.request_timeout_secs),
        ))
        .layer(RequestBodyLimitLayer::new(
            (config.max_body_size_mb as usize) * 1024 * 1024,
        ))
        .layer(from_fn(request_logging_middleware))
        .layer(from_fn_with_state(rate_limiter, rate_limit_middleware))
        .layer(from_fn_with_state(metrics, metrics_middleware))
        .layer(from_fn(crate::trace_context::trace_context_middleware))
        .layer(tower_http::request_id::SetRequestIdLayer::x_request_id(
            tower_http::request_id::MakeRequestUuid,
        ));

    router.with_state(state)
}

/// Check auth for API key management endpoints.
/// Allows access without auth when request comes from localhost (TUI admin)
/// or no keys exist yet (bootstrap mode). Otherwise requires a valid
/// multi-tenant key.
async fn check_auth_for_key_mgmt(
    headers: &HeaderMap,
    state: &AppState,
    client_ip: Option<std::net::IpAddr>,
) -> bool {
    // Localhost (TUI) can always manage keys — admin access
    if let Some(ip) = client_ip {
        if ip.is_loopback() {
            return true;
        }
    }
    check_auth_any(headers, state, client_ip).await
}

/// Check auth for model management endpoints (load/unload/set default).
/// Allows access without auth when:
/// - No keys are configured at all (bootstrap mode)
/// - Request comes from localhost (TUI admin)
///
/// Otherwise requires a valid multi-tenant key.
async fn check_auth_any(
    headers: &HeaderMap,
    state: &AppState,
    client_ip: Option<std::net::IpAddr>,
) -> bool {
    // If no key manager, allow all
    if state.api_key_manager.is_none() {
        return true;
    }

    // If key manager has no keys, allow all (bootstrap mode)
    if let Some(ref mgr_arc) = state.api_key_manager {
        let mgr = mgr_arc.lock().await;
        if mgr.list_keys().is_empty() {
            return true;
        }
    }

    // Localhost (TUI) can always manage models — admin access
    if let Some(ip) = client_ip {
        if ip.is_loopback() {
            return true;
        }
    }

    // Check multi-tenant keys
    if let Some(ref token) = extract_bearer(headers) {
        if let Some(ref mgr_arc) = state.api_key_manager {
            let mgr = mgr_arc.lock().await;
            if mgr.validate(token).is_some() {
                return true;
            }
        }
    }

    false
}

/// Extract the bearer token from the Authorization header.
fn extract_bearer(headers: &HeaderMap) -> Option<String> {
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                return Some(token.to_string());
            }
        }
    }
    None
}

/// Full authentication check: validates against the multi-tenant ApiKeyManager.
/// Returns an AuthResult indicating the outcome.
async fn check_auth_full(headers: &HeaderMap, state: &AppState, model: Option<&str>) -> AuthResult {
    // If no key manager, allow all
    if state.api_key_manager.is_none() {
        return AuthResult::NoAuthRequired;
    }

    // If key manager has no keys, allow all (bootstrap mode)
    if let Some(ref mgr_arc) = state.api_key_manager {
        let mgr = mgr_arc.lock().await;
        if mgr.list_keys().is_empty() {
            return AuthResult::NoAuthRequired;
        }
    }

    let token = match extract_bearer(headers) {
        Some(t) => t,
        None => return AuthResult::Unauthorized,
    };

    // Check API key manager
    if let Some(ref mgr_arc) = state.api_key_manager {
        let mut mgr = mgr_arc.lock().await;
        let key = match mgr.validate(&token) {
            Some(k) => k.clone(),
            None => return AuthResult::Unauthorized,
        };

        // Check model access
        if let Some(model) = model {
            if !mgr.check_model_access(&key, model) {
                return AuthResult::Forbidden;
            }
        }

        // Check rate limit
        if !mgr.check_rate_limit(&key) {
            return AuthResult::RateLimited;
        }

        // Check token quota
        if !mgr.check_token_quota(&key) {
            return AuthResult::QuotaExceeded;
        }

        return AuthResult::Allowed {
            key_id: key.key_id.clone(),
            key_name: key.name.clone(),
        };
    }

    AuthResult::Unauthorized
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let mgr = state.model_manager.lock().await;
    let uptime = state.start_time.elapsed().as_secs();
    let models = mgr.list();

    // Check if the backend is actually responsive by doing a quick
    // health check on the underlying llama-server. This detects cases
    // where the axum server is alive but the backend is deadlocked.
    let backend_healthy = if mgr.has_models() {
        if let Some(backend) = mgr.get(None) {
            // The backend's health check is backend-specific
            // (llama-server pings its own /health endpoint with a 3s timeout)
            backend.health_check().await.unwrap_or(false)
        } else {
            true // No backend to check
        }
    } else {
        true // No models loaded, nothing to check
    };

    let status = if backend_healthy { "ok" } else { "degraded" };

    let mut json = serde_json::json!({
        "status": status,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime,
        "models_loaded": mgr.count(),
        "default_model": mgr.default_id(),
        "backend_healthy": backend_healthy,
    });

    if !backend_healthy {
        json["warning"] = serde_json::json!(
            "Backend is not responding. The inference server may be deadlocked or out of memory."
        );
    }

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

/// Queue status endpoint — returns current queue depth and active requests.
/// Useful for monitoring and autoscaling decisions.
async fn queue_status(State(state): State<AppState>) -> impl IntoResponse {
    use std::sync::atomic::Ordering;

    let active = state.active_requests.load(Ordering::Relaxed);
    let waiting = state.waiting_requests.load(Ordering::Relaxed);
    let max_concurrent = state.semaphore.available_permits();

    Json(serde_json::json!({
        "active_requests": active,
        "waiting_requests": waiting,
        "available_permits": max_concurrent,
        "queue_visibility_enabled": state.queue_visibility,
    }))
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

async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
) -> Response {
    if !check_auth_any(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mgr = state.model_manager.lock().await;
    let models = mgr.list();
    let default_id = mgr.default_id().map(|s| s.to_string());
    let data: Vec<_> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": chrono::Utc::now().timestamp(),
                "owned_by": "athenas-studio",
                "backend": m.backend_name,
                "model_name": m.model_info.name,
                "context_size": m.model_info.context_size,
                "gpu_layers": m.model_info.gpu_layers,
                "is_default": default_id.as_deref() == Some(&m.id),
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
    /// Optional session ID for server-side conversation history.
    /// If provided, the server will prepend stored history to the messages.
    session_id: Option<String>,
    /// If true and session_id is provided, append the new messages to the session.
    #[serde(default = "default_true")]
    append_to_session: bool,
    /// If true and session_id is provided, save the assistant response to the session.
    #[serde(default = "default_true")]
    save_response: bool,
    /// Tools/functions for function calling (OpenAI format)
    tools: Option<serde_json::Value>,
    /// Tool choice: "auto", "none", "required", or specific tool object
    tool_choice: Option<serde_json::Value>,
    /// JSON mode / structured output: {"type": "json_object"} or {"type": "json_schema", ...}
    response_format: Option<serde_json::Value>,
    /// GBNF grammar string for grammar-constrained generation
    grammar: Option<String>,
}

fn default_true() -> bool {
    true
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

#[tracing::instrument(skip_all, fields(endpoint = "/v1/chat/completions"))]
async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model_for_auth = req.model.as_deref();
    let auth_key_id: Option<String> = match check_auth_full(&headers, &state, model_for_auth).await
    {
        AuthResult::NoAuthRequired => None,
        AuthResult::Allowed { key_id, .. } => Some(key_id),
        AuthResult::Unauthorized => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid or missing API key"})),
            )
                .into_response();
        }
        AuthResult::RateLimited => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Rate limit exceeded for this API key"})),
            )
                .into_response();
        }
        AuthResult::QuotaExceeded => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Daily token quota exceeded"})),
            )
                .into_response();
        }
        AuthResult::Forbidden => {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "This API key is not allowed to use the requested model"}))).into_response();
        }
    };

    let (_permit, queue_pos) = match acquire_with_tracking(&state).await {
        Some((permit, pos)) => (permit, pos),
        None => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };

    // ─── Semantic cache: check for a hit before inference ───
    // Only cache non-streaming requests (streaming cache is complex)
    let is_streaming = req.stream.unwrap_or(false);
    let cache_enabled = state
        .semantic_cache
        .as_ref()
        .map(|c| c.try_lock().map(|c| c.is_enabled()).unwrap_or(false))
        .unwrap_or(false);

    // Save cache-relevant fields before req is moved into chat_req
    let cache_model = req.model.clone();
    let cache_user_msg: String = if cache_enabled && !is_streaming {
        req.messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_text().to_string())
            .unwrap_or_default()
    } else {
        String::new()
    };

    if cache_enabled && !is_streaming && !cache_user_msg.is_empty() {
        // Generate embedding for the query
        let mgr = state.model_manager.lock().await;
        let backend_opt = mgr.get(cache_model.as_deref());
        drop(mgr);

        if let Some(backend) = backend_opt {
            let embed_req = EmbeddingRequest {
                model: cache_model.clone().unwrap_or_default(),
                input: athenas_inference::types::EmbeddingInput::Single(cache_user_msg.clone()),
                encoding_format: "float".to_string(),
            };

            if let Ok(embed_resp) = backend.embeddings(embed_req).await {
                if let Some(embed_data) = embed_resp.data.first() {
                    let query_embedding = embed_data.embedding.clone();

                    // Check cache
                    if let Some(ref cache_arc) = state.semantic_cache {
                        let mut cache = cache_arc.lock().await;
                        if let Some(cached) = cache.lookup(&query_embedding) {
                            tracing::info!(
                                "Semantic cache HIT for chat completion, tokens_saved={}",
                                cached.tokens_saved
                            );

                            // Record audit
                            record_audit(
                                &state,
                                AuditParams {
                                    endpoint: "/v1/chat/completions",
                                    method: "POST",
                                    status: 200,
                                    model: &cached.model,
                                    tokens_prompt: 0,
                                    tokens_generated: 0,
                                    error: None,
                                },
                            )
                            .await;

                            let cached_json: serde_json::Value =
                                serde_json::from_str(&cached.response_json)
                                    .unwrap_or(serde_json::Value::Null);

                            let mut resp = with_queue_headers(
                                &state,
                                queue_pos,
                                Json(cached_json).into_response(),
                            );
                            resp.headers_mut().insert("X-Cache", "HIT".parse().unwrap());
                            return resp;
                        }
                    }
                }
            }
        }
    }

    let new_messages: Vec<ChatMessage> = req
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

    // If session_id is provided, build messages from session history + new messages
    let (messages, session_id) = if let Some(ref sid) = req.session_id {
        let mut sm = state.session_manager.lock().await;
        // Auto-create session if it doesn't exist
        if sm.get(sid).is_none() {
            sm.create(Some(sid.clone()));
        }
        let session = sm.get(sid).unwrap();
        let built = session.build_messages(&new_messages);
        drop(sm);
        (built, Some(sid.clone()))
    } else {
        (new_messages.clone(), None)
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

    let chat_req = ChatRequest {
        model: req.model.unwrap_or_default(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens.or(req.max_completion_tokens),
        stream: req.stream.unwrap_or(false),
        stop,
        seed: req.seed,
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        response_format: req.response_format.clone(),
        grammar: req.grammar.clone(),
    };

    let model_id = chat_req.model.clone();

    // Determine the model sequence (primary + fallbacks) using the router
    let model_sequence = if let Some(ref router_arc) = state.model_router {
        let router = router_arc.lock().await;
        router.get_model_sequence(&model_id)
    } else {
        vec![model_id.clone()]
    };

    // Resolve a backend Arc under a SHORT lock, then release it so inference
    // never blocks other requests on the model manager mutex.
    //
    // Two passes: first respect the router's circuit breaker, then (to avoid
    // spurious 404s while a loaded model is in cooldown) retry ignoring it.
    let (backend, resolved_model_id, _tried_models) = {
        let mgr = state.model_manager.lock().await;

        let mut backend: Option<Arc<dyn Backend>> = None;
        let mut resolved = String::new();
        let mut tried = Vec::new();

        for ignore_health in [false, true] {
            for mid in &model_sequence {
                if !ignore_health {
                    tried.push(mid.clone());
                    if let Some(ref router_arc) = state.model_router {
                        let router = router_arc.lock().await;
                        if !router.is_healthy(mid) {
                            tracing::warn!("Model '{}' in cooldown, trying fallbacks", mid);
                            continue;
                        }
                    }
                }
                if let Some(b) = mgr.get(Some(mid.as_str())) {
                    backend = Some(b);
                    resolved = mid.clone();
                    break;
                }
            }
            if backend.is_some() {
                break;
            }
        }

        // If the explicitly requested model doesn't exist, serve the default
        // model instead of returning 404 (OpenAI clients often send arbitrary
        // model names; a loaded default model should answer).
        if backend.is_none() {
            if let Some(b) = mgr.get(None) {
                if let Some(def_id) = mgr.default_id() {
                    if !model_id.is_empty() {
                        tracing::warn!(
                            "Requested model '{}' not loaded; serving default '{}'",
                            model_id,
                            def_id
                        );
                    }
                    resolved = def_id.to_string();
                }
                backend = Some(b);
            }
        }

        match backend {
            Some(b) => (b, resolved, tried),
            None => {
                let available: Vec<_> = mgr.list().iter().map(|m| m.id.clone()).collect();
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("No model loaded. Tried: {:?}. Available models: {:?}",
                            tried, available)
                    })),
                )
                    .into_response();
            }
        }
    };
    // model_manager lock is now released — inference below runs lock-free.

    if chat_req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let model_name = backend
            .model_info()
            .map(|i| i.name.clone())
            .unwrap_or_default();

        // Append new messages to session at stream start
        if let Some(ref sid) = session_id {
            if req.append_to_session {
                let mut sm = state.session_manager.lock().await;
                if let Some(session) = sm.get_mut(sid) {
                    for msg in &new_messages {
                        session.append(msg.clone());
                    }
                }
            }
        }

        let metrics = state.metrics.clone();
        let session_mgr_clone = state.session_manager.clone();
        let session_id_clone = session_id.clone();
        let should_save_response = req.save_response;
        let api_key_mgr_clone = state.api_key_manager.clone();
        let auth_key_id_clone = auth_key_id.clone();
        // Spawn the inference on the resolved backend Arc — no manager lock is
        // held while streaming, so other requests are never blocked.
        tokio::spawn(async move {
            if let Err(e) = backend.chat_stream(chat_req, tx).await {
                tracing::error!("Stream error: {}", e);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            let mut full_text = String::new();
            while let Some(chunk) = rx.recv().await {
                if chunk.done {
                    if let Some(stats) = &chunk.stats {
                        metrics.record_tokens(&model_name, stats.tokens_prompt, stats.tokens_generated);

                        // Record token usage for API key
                        if let Some(ref key_id) = auth_key_id_clone {
                            if let Some(ref mgr_arc) = api_key_mgr_clone {
                                let mut mgr = mgr_arc.lock().await;
                                if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id) {
                                    mgr.record_usage(&key, stats.tokens_prompt as u64, stats.tokens_generated as u64);
                                }
                            }
                        }
                    }

                    // Save assistant response to session
                    if should_save_response {
                        if let Some(ref sid) = session_id_clone {
                            let mut sm = session_mgr_clone.lock().await;
                            if let Some(session) = sm.get_mut(sid) {
                                session.append(ChatMessage::assistant(full_text.clone()));
                            }
                        }
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
                full_text.push_str(&chunk.text);
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
        match backend.chat(chat_req.clone()).await {
            Ok(resp) => {
                state.metrics.record_tokens(
                    &resp.model,
                    resp.stats.tokens_prompt,
                    resp.stats.tokens_generated,
                );

                // Record token usage for API key
                if let Some(ref key_id) = auth_key_id {
                    if let Some(ref mgr_arc) = state.api_key_manager {
                        let mut mgr = mgr_arc.lock().await;
                        if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id)
                        {
                            mgr.record_usage(
                                &key,
                                resp.stats.tokens_prompt as u64,
                                resp.stats.tokens_generated as u64,
                            );
                        }
                    }
                }

                // Record audit entry
                record_audit(
                    &state,
                    AuditParams {
                        endpoint: "/v1/chat/completions",
                        method: "POST",
                        status: 200,
                        model: &resp.model,
                        tokens_prompt: resp.stats.tokens_prompt as u64,
                        tokens_generated: resp.stats.tokens_generated as u64,
                        error: None,
                    },
                )
                .await;

                // Save messages to session if session_id was provided
                if let Some(ref sid) = session_id {
                    if req.append_to_session {
                        let mut sm = state.session_manager.lock().await;
                        if let Some(session) = sm.get_mut(sid) {
                            for msg in &new_messages {
                                session.append(msg.clone());
                            }
                        }
                    }
                    if req.save_response {
                        let mut sm = state.session_manager.lock().await;
                        if let Some(session) = sm.get_mut(sid) {
                            session.append(resp.message.clone());
                        }
                    }
                }

                let mut message_json = serde_json::json!({
                    "role": "assistant",
                    "content": resp.message.content.as_text(),
                });
                if let Some(ref tool_calls) = resp.tool_calls {
                    message_json["tool_calls"] =
                        serde_json::to_value(tool_calls).unwrap_or(serde_json::Value::Null);
                    // If there are tool calls, content is typically null
                    if !resp.message.content.as_text().is_empty() {
                        // keep content if present
                    } else {
                        message_json["content"] = serde_json::Value::Null;
                    }
                }

                let finish_reason = resp.finish_reason.unwrap_or_else(|| {
                    if resp.tool_calls.is_some() {
                        "tool_calls".to_string()
                    } else {
                        "stop".to_string()
                    }
                });

                let json = serde_json::json!({
                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    "object": "chat.completion",
                    "created": chrono::Utc::now().timestamp(),
                    "model": resp.model,
                    "choices": [{
                        "index": 0,
                        "message": message_json,
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": resp.stats.tokens_prompt,
                        "completion_tokens": resp.stats.tokens_generated,
                        "total_tokens": resp.stats.tokens_prompt + resp.stats.tokens_generated,
                    },
                });

                // ─── Semantic cache: store the response ───
                if cache_enabled && !is_streaming && !cache_user_msg.is_empty() {
                    // Re-generate embedding for caching (we need it for the key)
                    let mgr = state.model_manager.lock().await;
                    let backend_opt = mgr.get(cache_model.as_deref());
                    drop(mgr);

                    if let Some(backend) = backend_opt {
                        let embed_req = EmbeddingRequest {
                            model: cache_model.clone().unwrap_or_default(),
                            input: athenas_inference::types::EmbeddingInput::Single(
                                cache_user_msg.clone(),
                            ),
                            encoding_format: "float".to_string(),
                        };

                        if let Ok(embed_resp) = backend.embeddings(embed_req).await {
                            if let Some(embed_data) = embed_resp.data.first() {
                                let tokens_saved =
                                    (resp.stats.tokens_prompt + resp.stats.tokens_generated) as u64;
                                let response_json =
                                    serde_json::to_string(&json).unwrap_or_default();

                                if let Some(ref cache_arc) = state.semantic_cache {
                                    let mut cache = cache_arc.lock().await;
                                    cache.insert(
                                        cache_user_msg.clone(),
                                        embed_data.embedding.clone(),
                                        response_json,
                                        resp.model.clone(),
                                        tokens_saved,
                                    );
                                }
                            }
                        }
                    }
                }

                let mut http_resp =
                    with_queue_headers(&state, queue_pos, Json(json).into_response());
                http_resp
                    .headers_mut()
                    .insert("X-Cache", "MISS".parse().unwrap());
                http_resp
            }
            Err(e) => {
                tracing::error!(
                    "Chat completion error with model '{}': {}",
                    resolved_model_id,
                    e
                );

                // Record failure in router
                if let Some(ref router_arc) = state.model_router {
                    let mut router = router_arc.lock().await;
                    router.record_failure(&resolved_model_id);
                }

                state
                    .metrics
                    .errors_total
                    .with_label_values(&["/v1/chat/completions", "inference"])
                    .inc();

                // Try fallback models
                let fallback_idx = model_sequence
                    .iter()
                    .position(|m| m == &resolved_model_id)
                    .map(|i| i + 1)
                    .unwrap_or(model_sequence.len());

                for next_model in &model_sequence[fallback_idx..] {
                    tracing::info!("Attempting fallback model: {}", next_model);
                    if let Some(ref router_arc) = state.model_router {
                        let router = router_arc.lock().await;
                        if !router.is_healthy(next_model) {
                            continue;
                        }
                    }
                    let next_backend = {
                        let mgr = state.model_manager.lock().await;
                        mgr.get(Some(next_model.as_str()))
                    };
                    if let Some(next_b) = next_backend {
                        let mut fb_req = chat_req.clone();
                        fb_req.model = next_model.clone();
                        match next_b.chat(fb_req).await {
                            Ok(resp) => {
                                if let Some(ref router_arc) = state.model_router {
                                    let mut router = router_arc.lock().await;
                                    router.record_success(next_model);
                                }
                                state.metrics.record_tokens(
                                    &resp.model,
                                    resp.stats.tokens_prompt,
                                    resp.stats.tokens_generated,
                                );
                                tracing::info!("Fallback model '{}' succeeded", next_model);

                                record_audit(
                                    &state,
                                    AuditParams {
                                        endpoint: "/v1/chat/completions",
                                        method: "POST",
                                        status: 200,
                                        model: &resp.model,
                                        tokens_prompt: resp.stats.tokens_prompt as u64,
                                        tokens_generated: resp.stats.tokens_generated as u64,
                                        error: Some(&format!(
                                            "fallback from {}",
                                            resolved_model_id
                                        )),
                                    },
                                )
                                .await;

                                let mut message_json = serde_json::json!({
                                    "role": "assistant",
                                    "content": resp.message.content.as_text(),
                                });
                                if let Some(ref tool_calls) = resp.tool_calls {
                                    message_json["tool_calls"] = serde_json::to_value(tool_calls)
                                        .unwrap_or(serde_json::Value::Null);
                                }
                                let finish_reason = resp.finish_reason.unwrap_or_else(|| {
                                    if resp.tool_calls.is_some() {
                                        "tool_calls".to_string()
                                    } else {
                                        "stop".to_string()
                                    }
                                });

                                let json = serde_json::json!({
                                    "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                                    "object": "chat.completion",
                                    "created": chrono::Utc::now().timestamp(),
                                    "model": resp.model,
                                    "choices": [{
                                        "index": 0,
                                        "message": message_json,
                                        "finish_reason": finish_reason,
                                    }],
                                    "usage": {
                                        "prompt_tokens": resp.stats.tokens_prompt,
                                        "completion_tokens": resp.stats.tokens_generated,
                                        "total_tokens": resp.stats.tokens_prompt + resp.stats.tokens_generated,
                                    },
                                });
                                return Json(json).into_response();
                            }
                            Err(fe) => {
                                tracing::error!(
                                    "Fallback model '{}' also failed: {}",
                                    next_model,
                                    fe
                                );
                                if let Some(ref router_arc) = state.model_router {
                                    let mut router = router_arc.lock().await;
                                    router.record_failure(next_model);
                                }
                            }
                        }
                    }
                }

                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("All models in chain failed. Primary error: {}", e),
                    })),
                )
                    .into_response()
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
    /// GBNF grammar string for grammar-constrained generation
    grammar: Option<String>,
    /// JSON mode / structured output
    response_format: Option<serde_json::Value>,
}

#[tracing::instrument(skip_all, fields(endpoint = "/v1/completions"))]
async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequestBody>,
) -> Response {
    let model_for_auth = req.model.as_deref();
    let auth_key_id: Option<String> = match check_auth_full(&headers, &state, model_for_auth).await
    {
        AuthResult::NoAuthRequired => None,
        AuthResult::Allowed { key_id, .. } => Some(key_id),
        AuthResult::Unauthorized => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid or missing API key"})),
            )
                .into_response();
        }
        AuthResult::RateLimited => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Rate limit exceeded"})),
            )
                .into_response();
        }
        AuthResult::QuotaExceeded => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Daily token quota exceeded"})),
            )
                .into_response();
        }
        AuthResult::Forbidden => {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({"error": "Model not allowed for this API key"})),
            )
                .into_response();
        }
    };

    let (_permit, queue_pos) = match acquire_with_tracking(&state).await {
        Some((permit, pos)) => (permit, pos),
        None => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
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
        grammar: req.grammar.clone(),
        response_format: req.response_format.clone(),
    };

    let model_id = comp_req.model.clone();

    // Resolve backend Arc under a SHORT lock, then release it so inference
    // never blocks other requests on the model manager mutex.
    let backend = {
        let mgr = state.model_manager.lock().await;

        // Try the requested model first, then fall back to the default.
        let backend = mgr.get(Some(model_id.as_str())).or_else(|| mgr.get(None));

        match backend {
            Some(b) => {
                let resolved = b
                    .model_info()
                    .map(|i| i.name.clone())
                    .unwrap_or_else(|| model_id.clone());
                if model_id != resolved && !model_id.is_empty() {
                    tracing::warn!(
                        "Completions: requested model '{}' not loaded; serving default '{}'",
                        model_id,
                        resolved
                    );
                }
                b
            }
            None => {
                let available: Vec<_> = mgr.list().iter().map(|m| m.id.clone()).collect();
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("No model loaded. Requested: '{}'. Available models: {:?}",
                            model_id, available)
                    })),
                )
                    .into_response();
            }
        }
    };
    // model_manager lock is now released — inference below runs lock-free.

    if comp_req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let model_name = backend
            .model_info()
            .map(|i| i.name.clone())
            .unwrap_or_default();

        let metrics = state.metrics.clone();
        let api_key_mgr_clone = state.api_key_manager.clone();
        let auth_key_id_clone = auth_key_id.clone();
        // Spawn the inference on the resolved backend Arc — no manager lock is
        // held while streaming, so other requests are never blocked.
        tokio::spawn(async move {
            if let Err(e) = backend.complete_stream(comp_req, tx).await {
                tracing::error!("Stream error: {}", e);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(chunk) = rx.recv().await {
                if chunk.done {
                    if let Some(stats) = &chunk.stats {
                        metrics.record_tokens(&model_name, stats.tokens_prompt, stats.tokens_generated);

                        // Record token usage for API key
                        if let Some(ref key_id) = auth_key_id_clone {
                            if let Some(ref mgr_arc) = api_key_mgr_clone {
                                let mut mgr = mgr_arc.lock().await;
                                if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id) {
                                    mgr.record_usage(&key, stats.tokens_prompt as u64, stats.tokens_generated as u64);
                                }
                            }
                        }
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

                // Record token usage for API key
                if let Some(ref key_id) = auth_key_id {
                    if let Some(ref mgr_arc) = state.api_key_manager {
                        let mut mgr = mgr_arc.lock().await;
                        if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id)
                        {
                            mgr.record_usage(
                                &key,
                                resp.stats.tokens_prompt as u64,
                                resp.stats.tokens_generated as u64,
                            );
                        }
                    }
                }

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
                with_queue_headers(&state, queue_pos, Json(json).into_response())
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
    if !check_auth_any(&headers, &state, None).await {
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

// ── Audio transcription endpoint (OpenAI-compatible) ──────────────────

/// Transcribe audio using a Whisper model.
///
/// This endpoint accepts multipart form data with:
/// - `file`: the audio file (wav, mp3, flac, etc.)
/// - `model`: the Whisper model name/path (GGUF)
/// - `language`: optional language hint (e.g. "en", "pt")
/// - `response_format`: optional output format (text, json, srt, vtt)
/// - `translate`: optional, translate to English
#[tracing::instrument(skip_all, fields(endpoint = "/v1/audio/transcriptions"))]
async fn transcribe_audio(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: axum::extract::Multipart,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut file_data: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut model_name: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;
    let mut translate: Option<bool> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" | "audio" => {
                filename = field.file_name().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(bytes) => file_data = Some(bytes.to_vec()),
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(
                                serde_json::json!({"error": format!("Failed to read file: {}", e)}),
                            ),
                        )
                            .into_response();
                    }
                }
            }
            "model" => {
                if let Ok(bytes) = field.bytes().await {
                    model_name = Some(String::from_utf8_lossy(&bytes).to_string());
                }
            }
            "language" => {
                if let Ok(bytes) = field.bytes().await {
                    language = Some(String::from_utf8_lossy(&bytes).to_string());
                }
            }
            "response_format" => {
                if let Ok(bytes) = field.bytes().await {
                    response_format = Some(String::from_utf8_lossy(&bytes).to_string());
                }
            }
            "translate" => {
                if let Ok(bytes) = field.bytes().await {
                    let val = String::from_utf8_lossy(&bytes).to_string();
                    translate = Some(val == "true" || val == "1");
                }
            }
            _ => {
                // Skip unknown fields
                let _ = field.bytes().await;
            }
        }
    }

    let audio_data = match file_data {
        Some(d) => d,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "No audio file uploaded. Use multipart form with 'file' field."})),
            )
                .into_response();
        }
    };

    let model = match model_name {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'model' field. Specify a Whisper model name or path."})),
            )
                .into_response();
        }
    };

    // Resolve model path
    let model_path = match resolve_model_path(&model, &state).await {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": e})),
            )
                .into_response();
        }
    };

    // Create WhisperBackend (auto-downloads whisper-cli if needed)
    let backend = match athenas_inference::WhisperBackend::new().await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": format!("Failed to init whisper backend: {}", e)}),
                ),
            )
                .into_response();
        }
    };

    let request = athenas_inference::TranscriptionRequest {
        audio_data,
        filename: filename.unwrap_or_else(|| "audio.wav".to_string()),
        model: model_path,
        language: language.filter(|l| l != "auto"),
        response_format: response_format.or(Some("json".to_string())),
        translate,
        max_len: None,
    };

    match backend.transcribe(&request).await {
        Ok(resp) => {
            let format = request.response_format.as_deref().unwrap_or("json");
            match format {
                "text" => (StatusCode::OK, resp.text).into_response(),
                "srt" => {
                    let mut srt = String::new();
                    for seg in &resp.segments {
                        srt.push_str(&format!(
                            "{}\n{} --> {}\n{}\n\n",
                            seg.id + 1,
                            format_srt_time(seg.start),
                            format_srt_time(seg.end),
                            seg.text
                        ));
                    }
                    (StatusCode::OK, [("content-type", "text/plain")], srt).into_response()
                }
                "vtt" => {
                    let mut vtt = String::from("WEBVTT\n\n");
                    for seg in &resp.segments {
                        vtt.push_str(&format!(
                            "{} --> {}\n{}\n\n",
                            format_vtt_time(seg.start),
                            format_vtt_time(seg.end),
                            seg.text
                        ));
                    }
                    (StatusCode::OK, [("content-type", "text/vtt")], vtt).into_response()
                }
                _ => Json(resp).into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("Transcription failed: {}", e)})),
        )
            .into_response(),
    }
}

/// Resolve a model name/ID to a file path on disk.
async fn resolve_model_path(model: &str, _state: &AppState) -> Result<String, String> {
    // If it's a direct path that exists, use it
    if std::path::Path::new(model).exists() {
        return Ok(model.to_string());
    }

    // Search in model registry
    let config = athenas_core::config::AppConfig::load()
        .map_err(|e| format!("Failed to load config: {}", e))?;
    let registry = athenas_core::model_registry::ModelRegistry::new(config.paths.models_dir);

    registry
        .find_model(model)
        .map(|m| m.file_path.to_string_lossy().to_string())
        .map_err(|e| format!("Model '{}' not found: {}", model, e))
}

fn format_srt_time(secs: f64) -> String {
    let hours = (secs / 3600.0) as u64;
    let mins = ((secs % 3600.0) / 60.0) as u64;
    let seconds = (secs % 60.0) as u64;
    let millis = ((secs % 1.0) * 1000.0) as u64;
    format!("{:02}:{:02}:{:02},{:03}", hours, mins, seconds, millis)
}

fn format_vtt_time(secs: f64) -> String {
    let hours = (secs / 3600.0) as u64;
    let mins = ((secs % 3600.0) / 60.0) as u64;
    let seconds = (secs % 60.0) as u64;
    let millis = ((secs % 1.0) * 1000.0) as u64;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, seconds, millis)
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
    mmproj_path: Option<String>,
    /// LoRA adapter file paths to load
    #[serde(default)]
    lora_paths: Vec<String>,
    /// Number of parallel slots for batched inference
    parallel_slots: Option<u32>,
    set_default: Option<bool>,
}

#[tracing::instrument(skip_all, fields(endpoint = "/v1/models/load"))]
async fn load_model_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, Some(addr.ip())).await {
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
        gpu_runtime: athenas_core::GpuRuntime::Auto,
        gpu_device: None,
        context_size: req.context_size.unwrap_or(4096),
        batch_size: req.batch_size.unwrap_or(512),
        threads: req.threads.unwrap_or(0),
        flash_attention: req.flash_attention.unwrap_or(true),
        use_mmap: true,
        use_mlock: false,
        reasoning_enabled: req.reasoning_enabled.unwrap_or(false),
        reasoning_budget: req.reasoning_budget.unwrap_or(-1),
        mmproj_path: req.mmproj_path,
        lora_paths: req.lora_paths.clone(),
        parallel_slots: req.parallel_slots.unwrap_or(4),
        draft_model_path: None,
        draft_max_tokens: 16,
        draft_min_ctx: 512,
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
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Json(req): Json<UnloadModelRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, Some(addr.ip())).await {
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

#[derive(Debug, Deserialize)]
struct SetDefaultModelRequest {
    model_id: String,
}

#[tracing::instrument(skip_all, fields(endpoint = "/v1/models/default"))]
async fn set_default_model_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Json(req): Json<SetDefaultModelRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut mgr = state.model_manager.lock().await;

    match mgr.set_default(&req.model_id) {
        Ok(()) => {
            let count = mgr.count();
            Json(serde_json::json!({
                "status": "ok",
                "default_model": req.model_id,
                "models_loaded": count,
            }))
            .into_response()
        }
        Err(e) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": e}))).into_response(),
    }
}

// ── Session management endpoints ─────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CreateSessionRequest {
    session_id: Option<String>,
    system_prompt: Option<String>,
    model: Option<String>,
}

async fn create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateSessionRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut sm = state.session_manager.lock().await;
    let id = sm.create(req.session_id);

    if let Some(sys) = req.system_prompt {
        if let Some(session) = sm.get_mut(&id) {
            session.system_prompt = Some(sys);
        }
    }
    if let Some(model) = req.model {
        if let Some(session) = sm.get_mut(&id) {
            session.model_id = Some(model);
        }
    }

    let info = sm.get(&id).map(|s| SessionInfo {
        id: s.id.clone(),
        model_id: s.model_id.clone(),
        message_count: s.messages.len(),
        slot_id: s.slot_id,
        slot_cache_warm: s.slot_cache_warm,
        created_at_secs: s.created_at.elapsed().as_secs(),
        last_active_secs: s.last_active.elapsed().as_secs(),
        system_prompt: s.system_prompt.clone(),
    });

    Json(serde_json::json!({
        "status": "created",
        "session": info,
    }))
    .into_response()
}

async fn list_sessions(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let sm = state.session_manager.lock().await;
    let sessions: Vec<SessionInfo> = sm.list();

    Json(serde_json::json!({
        "sessions": sessions,
        "total": sessions.len(),
    }))
    .into_response()
}

async fn get_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let sm = state.session_manager.lock().await;
    match sm.get(&id) {
        Some(session) => {
            let info = SessionInfo {
                id: session.id.clone(),
                model_id: session.model_id.clone(),
                message_count: session.messages.len(),
                slot_id: session.slot_id,
                slot_cache_warm: session.slot_cache_warm,
                created_at_secs: session.created_at.elapsed().as_secs(),
                last_active_secs: session.last_active.elapsed().as_secs(),
                system_prompt: session.system_prompt.clone(),
            };
            Json(serde_json::json!({"session": info})).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("Session '{}' not found", id)})),
        )
            .into_response(),
    }
}

async fn delete_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut sm = state.session_manager.lock().await;
    if sm.remove(&id) {
        Json(serde_json::json!({"status": "deleted", "session_id": id})).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("Session '{}' not found", id)})),
        )
            .into_response()
    }
}

async fn get_session_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let sm = state.session_manager.lock().await;
    match sm.get(&id) {
        Some(session) => {
            let messages: Vec<serde_json::Value> = session
                .messages
                .iter()
                .map(|m| {
                    let role = match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    };
                    serde_json::json!({
                        "role": role,
                        "content": m.content.as_text(),
                    })
                })
                .collect();

            Json(serde_json::json!({
                "session_id": id,
                "messages": messages,
                "count": messages.len(),
            }))
            .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("Session '{}' not found", id)})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct SetSystemPromptRequest {
    system_prompt: String,
}

async fn set_session_system_prompt(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(req): Json<SetSystemPromptRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut sm = state.session_manager.lock().await;
    match sm.get_mut(&id) {
        Some(session) => {
            session.system_prompt = Some(req.system_prompt);
            Json(serde_json::json!({"status": "updated", "session_id": id})).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("Session '{}' not found", id)})),
        )
            .into_response(),
    }
}

async fn purge_expired_sessions(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mut sm = state.session_manager.lock().await;
    let purged = sm.purge_expired();

    Json(serde_json::json!({
        "status": "purged",
        "sessions_removed": purged,
        "sessions_remaining": sm.count(),
    }))
    .into_response()
}

// ── Slot management endpoints ────────────────────────────────────────

async fn list_slots(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let Some(ref slot_mgr) = state.slot_manager else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({"error": "Slot manager not configured. Only available with llama-server backend."})),
        )
            .into_response();
    };

    match slot_mgr.list_slots().await {
        Ok(slots) => Json(serde_json::json!({"slots": slots})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct SaveSlotRequest {
    checkpoint_name: String,
}

async fn save_slot(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(slot_id): axum::extract::Path<i32>,
    Json(req): Json<SaveSlotRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let Some(ref slot_mgr) = state.slot_manager else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({"error": "Slot manager not configured"})),
        )
            .into_response();
    };

    match slot_mgr.save_slot(slot_id, &req.checkpoint_name).await {
        Ok(()) => Json(serde_json::json!({
            "status": "saved",
            "slot_id": slot_id,
            "checkpoint": req.checkpoint_name,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn restore_slot(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(slot_id): axum::extract::Path<i32>,
    Json(req): Json<SaveSlotRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let Some(ref slot_mgr) = state.slot_manager else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({"error": "Slot manager not configured"})),
        )
            .into_response();
    };

    match slot_mgr.restore_slot(slot_id, &req.checkpoint_name).await {
        Ok(()) => Json(serde_json::json!({
            "status": "restored",
            "slot_id": slot_id,
            "checkpoint": req.checkpoint_name,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn erase_slot(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(slot_id): axum::extract::Path<i32>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let Some(ref slot_mgr) = state.slot_manager else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({"error": "Slot manager not configured"})),
        )
            .into_response();
    };

    match slot_mgr.erase_slot(slot_id).await {
        Ok(()) => Json(serde_json::json!({
            "status": "erased",
            "slot_id": slot_id,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// ─── Embeddings ───

#[derive(Debug, Deserialize)]
struct EmbeddingsApiRequest {
    model: Option<String>,
    input: serde_json::Value,
    #[serde(default = "default_encoding_format_api")]
    encoding_format: Option<String>,
}

fn default_encoding_format_api() -> Option<String> {
    Some("float".to_string())
}

// ─── Tokenize endpoint ───

#[derive(Debug, Deserialize)]
struct TokenizeApiRequest {
    /// Text to tokenize
    text: String,
    /// If true, treat `text` as comma-separated token IDs to detokenize
    #[serde(default)]
    detokenize: bool,
}

#[tracing::instrument(skip_all, fields(endpoint = "/v1/tokenize"))]
async fn tokenize(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TokenizeApiRequest>,
) -> Response {
    // Auth check (same as embeddings)
    let auth_key_id: Option<String> = match check_auth_full(&headers, &state, None).await {
        AuthResult::NoAuthRequired => None,
        AuthResult::Allowed { key_id, .. } => Some(key_id),
        AuthResult::Unauthorized => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid or missing API key"})),
            )
                .into_response();
        }
        AuthResult::RateLimited => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Rate limit exceeded"})),
            )
                .into_response();
        }
        AuthResult::QuotaExceeded => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Daily token quota exceeded"})),
            )
                .into_response();
        }
        AuthResult::Forbidden => {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({"error": "Model not allowed for this API key"})),
            )
                .into_response();
        }
    };

    // Resolve backend
    let backend = {
        let mgr = state.model_manager.lock().await;
        mgr.get(None)
    };

    let Some(backend) = backend else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "No model loaded"})),
        )
            .into_response();
    };

    let result = backend
        .tokenize(TokenizeRequest {
            text: req.text,
            detokenize: req.detokenize,
        })
        .await;

    match result {
        Ok(resp) => {
            // Record usage for API key (tokenize uses prompt tokens)
            if let Some(ref key_id) = auth_key_id {
                if let Some(ref mgr_arc) = state.api_key_manager {
                    let mut mgr = mgr_arc.lock().await;
                    if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id) {
                        mgr.record_usage(&key, resp.count as u64, 0);
                    }
                }
            }
            Json(resp).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// ─── /v1/responses endpoint (OpenAI Responses API) ───

#[derive(Debug, Deserialize)]
struct ResponsesApiRequest {
    model: Option<String>,
    /// `input` can be a simple string or an array of messages (same format
    /// as chat completions `messages`).
    input: serde_json::Value,
    instructions: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_output_tokens: Option<u32>,
    stream: Option<bool>,
    stop: Option<serde_json::Value>,
    seed: Option<u64>,
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    response_format: Option<serde_json::Value>,
    grammar: Option<String>,
}

/// Convert the `input` field (string or array of messages) into ChatMessage list.
fn parse_responses_input(
    input: &serde_json::Value,
    instructions: &Option<String>,
) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    // Prepend system instructions if provided
    if let Some(ref instr) = instructions {
        if !instr.is_empty() {
            messages.push(ChatMessage::system(instr.clone()));
        }
    }

    match input {
        serde_json::Value::String(s) => {
            messages.push(ChatMessage::user(s.clone()));
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let Some(role) = item.get("role").and_then(|r| r.as_str()) {
                    let content = match item.get("content") {
                        Some(serde_json::Value::String(s)) => MessageContent::Text(s.clone()),
                        Some(serde_json::Value::Array(_)) => {
                            // Parse as content parts (multimodal)
                            serde_json::from_value(item["content"].clone())
                                .map(MessageContent::Parts)
                                .unwrap_or(MessageContent::Text(String::new()))
                        }
                        _ => MessageContent::Text(String::new()),
                    };
                    let role_enum = match role {
                        "system" => Role::System,
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        _ => Role::User,
                    };
                    messages.push(ChatMessage {
                        role: role_enum,
                        content,
                    });
                }
            }
        }
        _ => {
            // Fallback: treat as single user message
            messages.push(ChatMessage::user(input.to_string()));
        }
    }

    messages
}

#[tracing::instrument(skip_all, fields(endpoint = "/v1/responses"))]
async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ResponsesApiRequest>,
) -> Response {
    let model_for_auth = req.model.as_deref();
    let auth_key_id: Option<String> = match check_auth_full(&headers, &state, model_for_auth).await
    {
        AuthResult::NoAuthRequired => None,
        AuthResult::Allowed { key_id, .. } => Some(key_id),
        AuthResult::Unauthorized => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid or missing API key"})),
            )
                .into_response();
        }
        AuthResult::RateLimited => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Rate limit exceeded for this API key"})),
            )
                .into_response();
        }
        AuthResult::QuotaExceeded => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Daily token quota exceeded"})),
            )
                .into_response();
        }
        AuthResult::Forbidden => {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "This API key is not allowed to use the requested model"}))).into_response();
        }
    };

    let (_permit, queue_pos) = match acquire_with_tracking(&state).await {
        Some((permit, pos)) => (permit, pos),
        None => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };

    let messages = parse_responses_input(&req.input, &req.instructions);

    let stop = match &req.stop {
        Some(serde_json::Value::String(s)) => Some(vec![s.clone()]),
        Some(serde_json::Value::Array(arr)) => Some(
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
        ),
        _ => None,
    };

    let chat_req = ChatRequest {
        model: req.model.clone().unwrap_or_default(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_output_tokens,
        stream: req.stream.unwrap_or(false),
        stop,
        seed: req.seed,
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        response_format: req.response_format.clone(),
        grammar: req.grammar.clone(),
    };

    let model_id = chat_req.model.clone();

    // Resolve backend (same logic as chat_completions)
    let (backend, resolved_model_id, _tried_models): (Arc<dyn Backend>, String, Vec<String>) = {
        let mgr = state.model_manager.lock().await;

        let mut backend: Option<Arc<dyn Backend>> = None;
        let mut resolved = String::new();

        if let Some(b) = mgr.get(Some(model_id.as_str())) {
            backend = Some(b);
            resolved = model_id.clone();
        }
        if backend.is_none() {
            if let Some(b) = mgr.get(None) {
                if let Some(def_id) = mgr.default_id() {
                    resolved = def_id.to_string();
                }
                backend = Some(b);
            }
        }

        match backend {
            Some(b) => (b, resolved, vec![]),
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("No model loaded for model '{}'", model_id)
                    })),
                )
                    .into_response();
            }
        }
    };

    if chat_req.stream {
        // Streaming responses API format
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let model_name = backend
            .model_info()
            .map(|i| i.name.clone())
            .unwrap_or_default();
        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());

        let metrics = state.metrics.clone();
        let api_key_mgr_clone = state.api_key_manager.clone();
        let auth_key_id_clone = auth_key_id.clone();

        tokio::spawn(async move {
            if let Err(e) = backend.chat_stream(chat_req, tx).await {
                tracing::error!("Stream error: {}", e);
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            let mut full_text = String::new();
            while let Some(chunk) = rx.recv().await {
                if chunk.done {
                    if let Some(stats) = &chunk.stats {
                        metrics.record_tokens(&model_name, stats.tokens_prompt, stats.tokens_generated);
                        if let Some(ref key_id) = auth_key_id_clone {
                            if let Some(ref mgr_arc) = api_key_mgr_clone {
                                let mut mgr = mgr_arc.lock().await;
                                if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id) {
                                    mgr.record_usage(&key, stats.tokens_prompt as u64, stats.tokens_generated as u64);
                                }
                            }
                        }
                    }

                    // Final response.completed event
                    let json = serde_json::json!({
                        "id": response_id,
                        "object": "response.completed",
                        "created_at": chrono::Utc::now().timestamp(),
                        "model": model_name,
                        "output": [{
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": full_text}],
                        }],
                        "usage": {
                            "input_tokens": chunk.stats.as_ref().map(|s| s.tokens_prompt).unwrap_or(0),
                            "output_tokens": chunk.stats.as_ref().map(|s| s.tokens_generated).unwrap_or(0),
                            "total_tokens": chunk.stats.as_ref().map(|s| s.tokens_prompt + s.tokens_generated).unwrap_or(0),
                        },
                    });
                    yield Ok::<Event, std::convert::Infallible>(
                        Event::default().data(json.to_string())
                    );
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                full_text.push_str(&chunk.text);
                let json = serde_json::json!({
                    "id": response_id,
                    "object": "response.output_text.delta",
                    "delta": chunk.text,
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
        // Non-streaming responses API
        match backend.chat(chat_req).await {
            Ok(resp) => {
                state.metrics.record_tokens(
                    &resp.model,
                    resp.stats.tokens_prompt,
                    resp.stats.tokens_generated,
                );

                if let Some(ref key_id) = auth_key_id {
                    if let Some(ref mgr_arc) = state.api_key_manager {
                        let mut mgr = mgr_arc.lock().await;
                        if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id)
                        {
                            mgr.record_usage(
                                &key,
                                resp.stats.tokens_prompt as u64,
                                resp.stats.tokens_generated as u64,
                            );
                        }
                    }
                }

                record_audit(
                    &state,
                    AuditParams {
                        endpoint: "/v1/responses",
                        method: "POST",
                        status: 200,
                        model: &resp.model,
                        tokens_prompt: resp.stats.tokens_prompt as u64,
                        tokens_generated: resp.stats.tokens_generated as u64,
                        error: None,
                    },
                )
                .await;

                let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
                let output_text = resp.message.content.as_text();

                let json = serde_json::json!({
                    "id": response_id,
                    "object": "response",
                    "created_at": chrono::Utc::now().timestamp(),
                    "model": resp.model,
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": output_text}],
                    }],
                    "output_text": output_text,
                    "usage": {
                        "input_tokens": resp.stats.tokens_prompt,
                        "output_tokens": resp.stats.tokens_generated,
                        "total_tokens": resp.stats.tokens_prompt + resp.stats.tokens_generated,
                    },
                });
                with_queue_headers(&state, queue_pos, Json(json).into_response())
            }
            Err(e) => {
                tracing::error!(
                    "Responses API error with model '{}': {}",
                    resolved_model_id,
                    e
                );

                state
                    .metrics
                    .errors_total
                    .with_label_values(&["/v1/responses", "inference"])
                    .inc();

                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {"message": e.to_string(), "type": "server_error"}
                    })),
                )
                    .into_response()
            }
        }
    }
}

#[tracing::instrument(skip_all, fields(endpoint = "/v1/embeddings"))]
async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<EmbeddingsApiRequest>,
) -> Response {
    let model_for_auth = req.model.as_deref();
    let auth_key_id: Option<String> = match check_auth_full(&headers, &state, model_for_auth).await
    {
        AuthResult::NoAuthRequired => None,
        AuthResult::Allowed { key_id, .. } => Some(key_id),
        AuthResult::Unauthorized => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Invalid or missing API key"})),
            )
                .into_response();
        }
        AuthResult::RateLimited => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Rate limit exceeded"})),
            )
                .into_response();
        }
        AuthResult::QuotaExceeded => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({"error": "Daily token quota exceeded"})),
            )
                .into_response();
        }
        AuthResult::Forbidden => {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({"error": "Model not allowed for this API key"})),
            )
                .into_response();
        }
    };

    let (_permit, queue_pos) = match acquire_with_tracking(&state).await {
        Some((permit, pos)) => (permit, pos),
        None => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };

    let model_id = req.model.clone().unwrap_or_default();

    // Resolve backend Arc under a SHORT lock, then release it so inference
    // never blocks other requests on the model manager mutex.
    let (backend, resolved_model_id) = {
        let mgr = state.model_manager.lock().await;

        // Try the requested model first, then fall back to the default.
        let backend = mgr.get(Some(model_id.as_str())).or_else(|| mgr.get(None));

        match backend {
            Some(b) => {
                let resolved = b
                    .model_info()
                    .map(|i| i.name.clone())
                    .unwrap_or_else(|| model_id.clone());
                if model_id != resolved && !model_id.is_empty() {
                    tracing::warn!(
                        "Embeddings: requested model '{}' not loaded; serving default '{}'",
                        model_id,
                        resolved
                    );
                }
                (b, resolved)
            }
            None => {
                let available: Vec<_> = mgr.list().iter().map(|m| m.id.clone()).collect();
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": format!("No model loaded. Requested: '{}'. Available models: {:?}",
                            model_id, available)
                    })),
                )
                    .into_response();
            }
        }
    };
    // model_manager lock is now released — inference below runs lock-free.

    let embedding_input = match &req.input {
        serde_json::Value::String(s) => athenas_inference::EmbeddingInput::Single(s.clone()),
        serde_json::Value::Array(arr) => {
            let strings: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if strings.is_empty() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "Input array must not be empty"})),
                )
                    .into_response();
            }
            athenas_inference::EmbeddingInput::Batch(strings)
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Input must be a string or array of strings"})),
            )
                .into_response();
        }
    };

    let model_name = backend
        .model_info()
        .map(|i| i.name.clone())
        .unwrap_or_default();

    let emb_req = EmbeddingRequest {
        model: model_name,
        input: embedding_input,
        encoding_format: req.encoding_format.unwrap_or_else(|| "float".to_string()),
    };

    match backend.embeddings(emb_req).await {
        Ok(resp) => {
            state
                .metrics
                .record_tokens(&resolved_model_id, resp.usage.prompt_tokens, 0);

            // Record token usage for API key
            if let Some(ref key_id) = auth_key_id {
                if let Some(ref mgr_arc) = state.api_key_manager {
                    let mut mgr = mgr_arc.lock().await;
                    if let Some(key) = mgr.list_keys().into_iter().find(|k| &k.key_id == key_id) {
                        mgr.record_usage(&key, resp.usage.prompt_tokens as u64, 0);
                    }
                }
            }

            with_queue_headers(&state, queue_pos, Json(resp).into_response())
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

// ─── Multi-tenant API Key Management ───

#[derive(Debug, Deserialize)]
struct CreateKeyRequest {
    name: String,
    #[serde(default = "default_rate_limit")]
    rate_limit_per_minute: u32,
    #[serde(default = "default_token_limit")]
    daily_token_limit: u64,
    #[serde(default)]
    allowed_models: Vec<String>,
    metadata: Option<serde_json::Value>,
}

fn default_rate_limit() -> u32 {
    60
}

fn default_token_limit() -> u64 {
    0 // 0 = unlimited
}

async fn create_api_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Json(req): Json<CreateKeyRequest>,
) -> Response {
    if !check_auth_for_key_mgmt(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mgr_arc = match &state.api_key_manager {
        Some(m) => m,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": "Multi-tenant API keys not enabled"})),
            )
                .into_response();
        }
    };

    let mut mgr = mgr_arc.lock().await;
    let key = mgr.create_key(
        &req.name,
        req.rate_limit_per_minute,
        req.daily_token_limit,
        req.allowed_models,
        req.metadata,
    );
    Json(key).into_response()
}

async fn list_api_keys(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
) -> Response {
    if !check_auth_for_key_mgmt(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mgr_arc = match &state.api_key_manager {
        Some(m) => m,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": "Multi-tenant API keys not enabled"})),
            )
                .into_response();
        }
    };

    let mgr = mgr_arc.lock().await;
    let keys: Vec<serde_json::Value> = mgr
        .list_keys()
        .iter()
        .map(|k| {
            let usage = mgr.get_usage(&k.key_id);
            let (requests, tokens_prompt, tokens_generated, tokens_total, usage_date) =
                if let Some((_, u)) = usage {
                    (
                        u.requests,
                        u.tokens_prompt,
                        u.tokens_generated,
                        u.tokens_total(),
                        u.date.clone(),
                    )
                } else {
                    (0u64, 0u64, 0u64, 0u64, String::new())
                };
            let rate_limit_remaining = mgr.rate_limit_remaining(k);
            serde_json::json!({
                "key_id": k.key_id,
                "api_key": k.api_key,
                "name": k.name,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "active": k.active,
                "rate_limit_per_minute": k.rate_limit_per_minute,
                "daily_token_limit": k.daily_token_limit,
                "allowed_models": k.allowed_models,
                "metadata": k.metadata,
                "usage": {
                    "date": usage_date,
                    "requests": requests,
                    "tokens_prompt": tokens_prompt,
                    "tokens_generated": tokens_generated,
                    "tokens_total": tokens_total,
                    "rate_limit_remaining": rate_limit_remaining,
                },
            })
        })
        .collect();
    Json(serde_json::json!({"keys": keys})).into_response()
}

async fn get_api_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Path(id): Path<String>,
) -> Response {
    if !check_auth_for_key_mgmt(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mgr_arc = match &state.api_key_manager {
        Some(m) => m,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let mgr = mgr_arc.lock().await;
    match mgr.get_key(&id) {
        Some(key) => Json(serde_json::json!({
            "key_id": key.key_id,
            "api_key": key.api_key,
            "name": key.name,
            "created_at": key.created_at,
            "expires_at": key.expires_at,
            "active": key.active,
            "rate_limit_per_minute": key.rate_limit_per_minute,
            "daily_token_limit": key.daily_token_limit,
            "allowed_models": key.allowed_models,
            "metadata": key.metadata,
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Key not found"})),
        )
            .into_response(),
    }
}

async fn delete_api_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Path(id): Path<String>,
) -> Response {
    if !check_auth_for_key_mgmt(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mgr_arc = match &state.api_key_manager {
        Some(m) => m,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let mut mgr = mgr_arc.lock().await;
    if mgr.delete_key(&id) {
        Json(serde_json::json!({"status": "deleted", "key_id": id})).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Key not found"})),
        )
            .into_response()
    }
}

async fn revoke_api_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Path(id): Path<String>,
) -> Response {
    if !check_auth_for_key_mgmt(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mgr_arc = match &state.api_key_manager {
        Some(m) => m,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let mut mgr = mgr_arc.lock().await;
    if mgr.revoke_key(&id) {
        Json(serde_json::json!({"status": "revoked", "key_id": id})).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Key not found"})),
        )
            .into_response()
    }
}

async fn get_api_key_usage(
    State(state): State<AppState>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<std::net::SocketAddr>,
    Path(id): Path<String>,
) -> Response {
    if !check_auth_for_key_mgmt(&headers, &state, Some(addr.ip())).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let mgr_arc = match &state.api_key_manager {
        Some(m) => m,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let mgr = mgr_arc.lock().await;
    match mgr.get_usage(&id) {
        Some((key, usage)) => {
            let remaining = mgr.rate_limit_remaining(key);
            Json(serde_json::json!({
                "key_id": key.key_id,
                "name": key.name,
                "date": usage.date,
                "requests": usage.requests,
                "tokens_prompt": usage.tokens_prompt,
                "tokens_generated": usage.tokens_generated,
                "tokens_total": usage.tokens_total(),
                "daily_token_limit": key.daily_token_limit,
                "rate_limit_per_minute": key.rate_limit_per_minute,
                "rate_limit_remaining": remaining,
            }))
            .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Key not found"})),
        )
            .into_response(),
    }
}

// ─── Model Routing & Fallback Chains ───

#[derive(Debug, Deserialize)]
struct CreateAliasRequest {
    alias: String,
    target: String,
}

async fn create_alias(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateAliasRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let router_arc = match &state.model_router {
        Some(r) => r,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let mut router = router_arc.lock().await;
    router.add_alias(&req.alias, &req.target);
    Json(serde_json::json!({"status": "created", "alias": req.alias, "target": req.target}))
        .into_response()
}

async fn list_aliases(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let router_arc = match &state.model_router {
        Some(r) => r,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let router = router_arc.lock().await;
    let aliases: Vec<serde_json::Value> = router
        .list_aliases()
        .iter()
        .map(|(a, t)| serde_json::json!({"alias": a, "target": t}))
        .collect();
    Json(serde_json::json!({"aliases": aliases})).into_response()
}

#[derive(Debug, Deserialize)]
struct CreateChainRequest {
    primary: String,
    fallbacks: Vec<String>,
    max_retries: Option<u32>,
    timeout_secs: Option<u64>,
}

async fn create_chain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateChainRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let router_arc = match &state.model_router {
        Some(r) => r,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let mut router = router_arc.lock().await;
    let chain = crate::model_router::FallbackChain {
        primary: req.primary.clone(),
        fallbacks: req.fallbacks.clone(),
        max_retries: req.max_retries.unwrap_or(1),
        timeout_secs: req.timeout_secs.unwrap_or(0),
    };
    router.add_chain(chain);
    Json(serde_json::json!({
        "status": "created",
        "primary": req.primary,
        "fallbacks": req.fallbacks,
    }))
    .into_response()
}

async fn list_chains(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let router_arc = match &state.model_router {
        Some(r) => r,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let router = router_arc.lock().await;
    let chains = router.list_chains();
    Json(serde_json::json!({"chains": chains})).into_response()
}

async fn routing_health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let router_arc = match &state.model_router {
        Some(r) => r,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let router = router_arc.lock().await;
    let health = router.health_status();
    Json(serde_json::json!({"models": health})).into_response()
}

// ─── Audit Logging ───

/// Parameters for recording an audit entry.
struct AuditParams<'a> {
    endpoint: &'a str,
    method: &'a str,
    status: u16,
    model: &'a str,
    tokens_prompt: u64,
    tokens_generated: u64,
    error: Option<&'a str>,
}

/// Helper to record an audit entry if the audit logger is enabled.
async fn record_audit(state: &AppState, params: AuditParams<'_>) {
    if let Some(ref logger_arc) = state.audit_logger {
        let entry = AuditEntry {
            id: format!("audit_{}", &uuid::Uuid::new_v4().to_string()[..12]),
            timestamp: chrono::Utc::now(),
            endpoint: params.endpoint.to_string(),
            method: params.method.to_string(),
            status: params.status,
            key_id: None,
            key_name: None,
            model: if params.model.is_empty() {
                None
            } else {
                Some(params.model.to_string())
            },
            tokens_prompt: params.tokens_prompt,
            tokens_generated: params.tokens_generated,
            latency_ms: 0,
            client_ip: None,
            user_agent: None,
            error: params.error.map(|e| e.to_string()),
            request_id: None,
        };
        let mut logger = logger_arc.lock().await;
        logger.record(entry);
    }
}

#[derive(Debug, Deserialize)]
struct AuditQueryParams {
    limit: Option<usize>,
    key_id: Option<String>,
    endpoint: Option<String>,
    min_status: Option<u16>,
}

async fn query_audit_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<AuditQueryParams>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let logger_arc = match &state.audit_logger {
        Some(l) => l,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let logger = logger_arc.lock().await;
    let entries = logger.query(
        params.limit.unwrap_or(100),
        params.key_id.as_deref(),
        params.endpoint.as_deref(),
        params.min_status,
    );
    Json(serde_json::json!({"logs": entries, "count": entries.len()})).into_response()
}

async fn audit_stats(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let logger_arc = match &state.audit_logger {
        Some(l) => l,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let logger = logger_arc.lock().await;
    let stats = logger.stats();
    Json(stats).into_response()
}

// ─── Vector Store Endpoints ───

#[derive(Debug, Deserialize)]
struct VsAddRequest {
    id: String,
    content: String,
    embedding: Vec<f32>,
    metadata: Option<serde_json::Value>,
}

async fn vs_add_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<VsAddRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": "Vector store not enabled"})),
            )
                .into_response()
        }
    };

    match vs
        .add_document(req.id, req.content, req.embedding, req.metadata)
        .await
    {
        Ok(doc) => Json(doc).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct VsAddBatchRequest {
    documents: Vec<VsAddRequest>,
}

async fn vs_add_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<VsAddBatchRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": "Vector store not enabled"})),
            )
                .into_response()
        }
    };

    let docs: Vec<(String, String, Vec<f32>, Option<serde_json::Value>)> = req
        .documents
        .into_iter()
        .map(|d| (d.id, d.content, d.embedding, d.metadata))
        .collect();

    match vs.add_documents(docs).await {
        Ok(added) => {
            Json(serde_json::json!({"added": added.len(), "documents": added})).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct VsSearchRequest {
    embedding: Vec<f32>,
    top_k: Option<usize>,
}

async fn vs_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<VsSearchRequest>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": "Vector store not enabled"})),
            )
                .into_response()
        }
    };

    let top_k = req.top_k.unwrap_or(5);
    let results = vs.search(&req.embedding, top_k).await;
    Json(serde_json::json!({"results": results, "count": results.len()})).into_response()
}

async fn vs_list_documents(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({"error": "Vector store not enabled"})),
            )
                .into_response()
        }
    };

    let docs = vs.list_documents().await;
    Json(serde_json::json!({"documents": docs, "count": docs.len()})).into_response()
}

async fn vs_get_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    match vs.get_document(&id).await {
        Some(doc) => Json(doc).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Document not found"})),
        )
            .into_response(),
    }
}

async fn vs_delete_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    if vs.delete_document(&id).await {
        Json(serde_json::json!({"status": "deleted", "id": id})).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Document not found"})),
        )
            .into_response()
    }
}

async fn vs_clear(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    vs.clear().await;
    Json(serde_json::json!({"status": "cleared"})).into_response()
}

async fn vs_stats(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !check_auth_any(&headers, &state, None).await {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let vs = match &state.vector_store {
        Some(v) => v,
        None => return StatusCode::NOT_IMPLEMENTED.into_response(),
    };

    let count = vs.len().await;
    Json(serde_json::json!({"document_count": count, "enabled": vs.is_enabled()})).into_response()
}

// ─── Semantic Cache Endpoints ───

#[tracing::instrument(skip_all, fields(endpoint = "/v1/cache/stats"))]
async fn cache_stats(State(state): State<AppState>) -> Response {
    match &state.semantic_cache {
        Some(cache) => {
            let c = cache.lock().await;
            let stats = c.stats();
            Json(serde_json::json!({
                "enabled": true,
                "entries": stats.entries,
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate(),
                "tokens_saved": stats.tokens_saved,
                "evictions": stats.evictions,
            }))
            .into_response()
        }
        None => Json(serde_json::json!({
            "enabled": false,
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "tokens_saved": 0,
            "evictions": 0,
        }))
        .into_response(),
    }
}

#[tracing::instrument(skip_all, fields(endpoint = "/v1/cache/clear"))]
async fn cache_clear(State(state): State<AppState>) -> Response {
    match &state.semantic_cache {
        Some(cache) => {
            let mut c = cache.lock().await;
            c.clear();
            Json(serde_json::json!({"status": "cleared"})).into_response()
        }
        None => (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({"error": "Semantic cache not enabled"})),
        )
            .into_response(),
    }
}
