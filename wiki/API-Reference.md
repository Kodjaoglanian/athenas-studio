# API Reference

The Athenas Studio API server is an OpenAI-compatible REST API. Start it with:

```bash
athenas serve model.gguf --port 8080
```

## Authentication

If `api_key` is set in the config (`[server] api_key = "..."`), all requests must include:

```
Authorization: Bearer <api_key>
```

For multi-tenant API keys, see [Multi-tenant API Keys](Multi-tenant-API-Keys).

## Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible chat completions endpoint with SSE streaming support.

**Request:**

```json
{
  "model": "llama-2-7b-chat",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048,
  "stream": false,
  "stop": ["</s>"],
  "session_id": "sess_abc123"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | default model | Model ID to use |
| `messages` | array | required | Chat messages |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling |
| `max_tokens` | int | 2048 | Max tokens to generate |
| `stream` | bool | false | Enable SSE streaming |
| `stop` | array | `[]` | Stop sequences |
| `session_id` | string | null | Session ID for conversation history |

**Non-streaming response:**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-2-7b-chat",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Streaming response (SSE):**

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"llama-2-7b-chat","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"llama-2-7b-chat","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"llama-2-7b-chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}

data: [DONE]
```

### Text Completions

```
POST /v1/completions
```

OpenAI-compatible text completions.

**Request:**

```json
{
  "model": "llama-2-7b-chat",
  "prompt": "Once upon a time",
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": false
}
```

### Embeddings

```
POST /v1/embeddings
```

Generate vector embeddings from text. See [Embeddings API](Embeddings-API) for details.

### List Models

```
GET /v1/models
```

Returns all loaded models.

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-2-7b-chat",
      "object": "model",
      "owned_by": "athenas"
    }
  ]
}
```

### Load Model

```
POST /v1/models/load
```

Load an additional model at runtime (multi-model support).

```json
{
  "model_path": "/path/to/second-model.gguf",
  "gpu_layers": -1,
  "context_size": 4096,
  "mmproj_path": "/path/to/mmproj.gguf",
  "set_default": false
}
```

### Unload Model

```
POST /v1/models/unload
```

Unload a model by ID to free memory.

```json
{
  "model_id": "model-1"
}
```

### File Upload

```
POST /v1/files
```

Upload files for multimodal inference (images, documents).

```bash
curl http://127.0.0.1:8080/v1/files \
  -F "file=@photo.jpg"
```

Response:

```json
{
  "id": "file-abc123",
  "filename": "photo.jpg",
  "bytes": 102400,
  "purpose": "vision"
}
```

Use the uploaded file in chat completions:

```json
{
  "model": "llama-3.2-vision",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "file:photo.jpg"}}
  ]}]
}
```

### Health Check

```
GET /v1/health
```

Returns server health, model info, uptime, and backend status.

```json
{
  "status": "ok",
  "model": "llama-2-7b-chat",
  "backend": "llama.cpp",
  "uptime_secs": 3600,
  "loaded_models": 1
}
```

### Kubernetes Readiness

```
GET /v1/ready
```

Returns `200` if a model is loaded, `503` if no model is loaded. Suitable for Kubernetes readiness probes.

### Health (alias)

```
GET /health
```

Alias for `/v1/health`.

### Metrics

```
GET /metrics
```

Prometheus-compatible metrics endpoint. Exposes:

- `athenas_requests_total` — Total requests by method/path/status
- `athenas_requests_active` — Currently active requests
- `athenas_request_duration_seconds` — Request latency histogram
- `athenas_tokens_prompt_total` — Total prompt tokens processed
- `athenas_tokens_generated_total` — Total tokens generated
- `athenas_errors_total` — Total errors by type

### Session Management

```
POST /v1/sessions
```

Create a new conversation session.

```json
{
  "session_id": "my-session",
  "system_prompt": "You are a helpful assistant."
}
```

```
GET /v1/sessions
```

List all active sessions.

```
GET /v1/sessions/{id}
```

Get session details including message history.

```
DELETE /v1/sessions/{id}
```

Delete a session and free its slot.

### Slot Management

```
GET /v1/slots
```

List all llama-server slots with KV cache state.

```
POST /v1/slots/{id}/save
```

Save a slot's KV cache to a named checkpoint.

```json
{"name": "checkpoint-1"}
```

```
POST /v1/slots/{id}/restore
```

Restore a slot's KV cache from a checkpoint.

```
POST /v1/slots/{id}
```

Erase a slot's KV cache (free memory).

```json
{"erase": true}
```

See [Session & Slot Management](Session-and-Slot-Management) for details.
