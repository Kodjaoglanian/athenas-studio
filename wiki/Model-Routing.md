# Model Routing & Fallback Chains

Model routing and fallback chains provide high availability and automatic failover for production inference workloads. When a model or backend fails, the server automatically retries with a configured fallback model.

## Overview

- **Routing rules**: Route requests to specific models based on request properties (model name, token count, task type)
- **Fallback chains**: Define ordered lists of models to try if the primary model fails
- **Automatic failover**: If a model is unloaded, crashes, or times out, the next model in the chain is tried
- **Health-based routing**: Skip unhealthy models automatically
- **Load balancing**: Distribute requests across multiple instances of the same model

## Configuration

Routing and fallback chains are configured in `~/.athenas/data/routing.json`:

```json
{
  "routes": [
    {
      "name": "default-fallback",
      "primary": "llama-3.1-8b-instruct",
      "fallbacks": ["qwen2.5-7b-instruct", "mistral-7b-instruct"],
      "timeout_ms": 30000,
      "retry_count": 1
    },
    {
      "name": "heavy-tasks",
      "match": {
        "min_tokens": 2000
      },
      "primary": "qwen2.5-32b-instruct",
      "fallbacks": ["llama-3.1-70b-instruct", "llama-3.1-8b-instruct"],
      "timeout_ms": 60000,
      "retry_count": 2
    }
  ],
  "default_route": "default-fallback",
  "health_check_interval_secs": 30
}
```

### Route Structure

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Route name (used in `default_route`) |
| `match` | object | Optional matching criteria |
| `match.model` | string | Match by model name in request |
| `match.min_tokens` | int | Match if `max_tokens` >= value |
| `match.max_tokens` | int | Match if `max_tokens` <= value |
| `primary` | string | Primary model ID |
| `fallbacks` | array | Ordered list of fallback model IDs |
| `timeout_ms` | int | Per-model timeout in milliseconds |
| `retry_count` | int | Number of retries per model before failing over |

### Matching Logic

Routes are evaluated in order. The first matching route is used. If no route matches, the `default_route` is used.

- If `match` is omitted, the route matches all requests
- If `match.model` is specified, it matches requests with that exact `model` field
- If `match.min_tokens` is specified, it matches requests where `max_tokens >= min_tokens`

## CLI Management

### Show Routing Config

```bash
athenas routing show
```

### Add a Route

```bash
athenas routing add \
  --name "production" \
  --primary "llama-3.1-8b-instruct" \
  --fallbacks "qwen2.5-7b-instruct,mistral-7b-instruct" \
  --timeout 30000 \
  --retries 1
```

### Remove a Route

```bash
athenas routing remove "production"
```

### Set Default Route

```bash
athenas routing default "production"
```

### Test Routing

```bash
athenas routing test --model "llama-3.1-8b" --max-tokens 4000
```

## API Management

### Get Routing Config

```
GET /v1/routing
Authorization: Bearer <admin-key>
```

### Update Routing Config

```
PUT /v1/routing
Authorization: Bearer <admin-key>
```

```json
{
  "routes": [...],
  "default_route": "default-fallback"
}
```

### Get Route Health

```
GET /v1/routing/health
Authorization: Bearer <admin-key>
```

Response:

```json
{
  "models": {
    "llama-3.1-8b-instruct": {
      "healthy": true,
      "last_check": "2024-01-15T10:00:00Z",
      "avg_latency_ms": 450,
      "error_rate": 0.01
    },
    "qwen2.5-7b-instruct": {
      "healthy": true,
      "last_check": "2024-01-15T10:00:00Z",
      "avg_latency_ms": 380,
      "error_rate": 0.0
    }
  }
}
```

## How Fallback Works

1. Request arrives at `/v1/chat/completions` or `/v1/completions`
2. The routing middleware evaluates routes to find the primary model
3. The request is sent to the primary model with the configured timeout
4. If the primary model fails (error, timeout, or model not loaded), the next fallback is tried
5. This continues until a model succeeds or all fallbacks are exhausted
6. If all models fail, a `503 Service Unavailable` is returned with details of which models were tried

### Response Headers

When a fallback is used, the response includes:

```
X-Athenas-Model-Used: qwen2.5-7b-instruct
X-Athenas-Fallback-From: llama-3.1-8b-instruct
X-Athenas-Fallback-Reason: timeout
```

## Example Scenarios

### Scenario 1: Primary Model Crashes

1. Request for `llama-3.1-8b-instruct` arrives
2. The model's llama-server process has crashed
3. Router detects the failure and tries `qwen2.5-7b-instruct`
4. Request succeeds with Qwen model
5. Response includes `X-Athenas-Fallback-From: llama-3.1-8b-instruct`

### Scenario 2: Timeout on Heavy Model

1. Request with `max_tokens: 4000` matches the "heavy-tasks" route
2. Primary `qwen2.5-32b-instruct` takes too long (exceeds 60s timeout)
3. Router falls back to `llama-3.1-70b-instruct`
4. If that also times out, falls back to `llama-3.1-8b-instruct`
5. The smaller model succeeds quickly

### Scenario 3: Model Not Loaded

1. Request arrives for a model that was unloaded
2. Router skips to the first available fallback
3. If no fallbacks are loaded, returns `503`

## Best Practices

- **Order fallbacks by capability** — Put the most similar model first
- **Include a small/fast model** as the last fallback — Ensures some response is always possible
- **Set reasonable timeouts** — 30s for chat, 60s for long generation
- **Monitor health** — Use `/v1/routing/health` to detect degraded models
- **Keep fallback models loaded** — Pre-load fallback models to avoid cold-start delays
- **Use matching for task routing** — Route heavy tasks to larger models, simple tasks to smaller ones
