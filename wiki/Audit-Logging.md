# Audit Logging

Athenas Studio provides comprehensive audit logging for compliance, security analysis, and usage tracking. All API requests are logged with full details including the caller identity, model used, token counts, and response status.

## Overview

- **JSONL format** — Structured logs for easy parsing and ingestion into SIEM systems
- **Server-side middleware** — Every API request is automatically logged
- **TUI logs page** — Real-time log viewing in the terminal (F6)
- **Configurable retention** — Automatic log rotation and cleanup
- **Sensitive data redaction** — API keys and request bodies can be redacted

## Configuration

In `~/.athenas/config.toml`:

```toml
[logging]
level = "info"                    # trace, debug, info, warn, error
file_logging = false              # write logs to file

[audit]
enabled = true                    # enable audit logging
log_file = "~/.athenas/data/audit.log"  # JSONL audit log path
max_file_size_mb = 100            # max file size before rotation
max_files = 10                    # max rotated files to keep
redact_api_keys = true            # redact API keys in logs
redact_request_body = false       # redact request bodies (may lose context)
include_response_summary = true   # include response status and token counts
```

## Audit Log Format

Each line in the audit log is a JSON object:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "event_type": "api_request",
  "method": "POST",
  "path": "/v1/chat/completions",
  "status": 200,
  "client_ip": "192.168.1.100",
  "api_key_id": "key_abc123",
  "api_key_name": "Engineering Team",
  "model": "llama-3.1-8b-instruct",
  "model_used": "llama-3.1-8b-instruct",
  "fallback_from": null,
  "fallback_reason": null,
  "tokens_prompt": 150,
  "tokens_generated": 85,
  "tokens_total": 235,
  "duration_ms": 1250,
  "session_id": "sess_xyz789",
  "stream": true,
  "user_agent": "openai-python/1.0"
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | Request timestamp with milliseconds |
| `event_type` | string | Event type (`api_request`, `model_load`, `model_unload`, `key_create`, `key_revoke`) |
| `method` | string | HTTP method |
| `path` | string | Request path |
| `status` | int | HTTP response status code |
| `client_ip` | string | Client IP address |
| `api_key_id` | string | API key ID used (if multi-tenant) |
| `api_key_name` | string | API key name (if multi-tenant) |
| `model` | string | Requested model |
| `model_used` | string | Actual model that served the request (may differ if fallback) |
| `fallback_from` | string or null | Original model if fallback was used |
| `fallback_reason` | string or null | Reason for fallback (`timeout`, `error`, `not_loaded`) |
| `tokens_prompt` | int | Prompt tokens processed |
| `tokens_generated` | int | Tokens generated |
| `tokens_total` | int | Total tokens |
| `duration_ms` | int | Request duration in milliseconds |
| `session_id` | string or null | Session ID if used |
| `stream` | bool | Whether streaming was used |
| `user_agent` | string | Client User-Agent header |

## Event Types

### `api_request`

Logged for every API request (chat completions, completions, embeddings, file uploads, etc.).

### `model_load`

Logged when a model is loaded (at startup or via `/v1/models/load`).

```json
{
  "timestamp": "2024-01-15T10:00:00.000Z",
  "event_type": "model_load",
  "model": "llama-3.1-8b-instruct",
  "model_path": "/path/to/model.gguf",
  "backend": "llama.cpp",
  "gpu_layers": -1,
  "context_size": 4096,
  "api_key_id": "key_admin",
  "duration_ms": 15000
}
```

### `model_unload`

Logged when a model is unloaded.

### `key_create`

Logged when a new API key is created (requires admin key).

### `key_revoke`

Logged when an API key is revoked.

## TUI Logs Page

Press **F6** in the TUI to view real-time application logs. The logs page shows:

- **Timestamp** — Time of the log entry
- **Level** — Log level (INFO, WARN, ERROR, DEBUG, TRACE)
- **Target** — Module that produced the log
- **Message** — Log message

### TUI Logs Controls

| Key | Action |
|-----|--------|
| **Up/Down** | Scroll through logs |
| **C** | Clear log buffer |
| **Auto-scroll** | Automatically follows latest entries |

The TUI log buffer holds up to 500 entries. For persistent logging, enable file logging in the config.

## Log Rotation

Audit logs are automatically rotated when they reach `max_file_size_mb`:

- `audit.log` → `audit.log.1` → `audit.log.2` → ... → `audit.log.10`
- Old rotated files are deleted when `max_files` is exceeded
- Rotation happens transparently — no server restart needed

## Querying Audit Logs

### Using `jq`

```bash
# All requests today
cat ~/.athenas/data/audit.log | jq 'select(.timestamp >= "2024-01-15")'

# Requests by a specific API key
cat ~/.athenas/data/audit.log | jq 'select(.api_key_id == "key_abc123")'

# Failed requests
cat ~/.athenas/data/audit.log | jq 'select(.status >= 400)'

# Token usage by model
cat ~/.athenas/data/audit.log | \
  jq -s 'group_by(.model_used) | map({model: .[0].model_used, total_tokens: map(.tokens_total) | add})'

# Fallback events
cat ~/.athenas/data/audit.log | jq 'select(.fallback_from != null)'
```

### Using the API

```
GET /v1/audit/logs?limit=100&event_type=api_request
Authorization: Bearer <admin-key>
```

Query parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `limit` | 100 | Max entries to return |
| `event_type` | all | Filter by event type |
| `api_key_id` | all | Filter by API key |
| `model` | all | Filter by model |
| `since` | all | ISO 8601 timestamp (inclusive) |
| `until` | all | ISO 8601 timestamp (exclusive) |

### Usage Summary

```
GET /v1/audit/summary?period=daily
Authorization: Bearer <admin-key>
```

Response:

```json
{
  "period": "daily",
  "summary": [
    {
      "date": "2024-01-15",
      "total_requests": 5420,
      "total_tokens": 1250000,
      "unique_keys": 8,
      "models_used": ["llama-3.1-8b", "qwen2.5-7b"],
      "error_count": 12,
      "fallback_count": 3
    }
  ]
}
```

## Integration with SIEM

The JSONL format is compatible with most SIEM and log aggregation systems:

- **Elasticsearch/Logstash/Kibana** — Use Filebeat to ship JSONL to ELK
- **Splunk** — Use the Splunk Universal Forwarder with JSON source type
- **Grafana/Loki** — Use Promtail to ingest JSONL logs
- **Datadog** — Use the Datadog Agent with JSON log collection

### Example: Filebeat Configuration

```yaml
filebeat.inputs:
- type: filestream
  paths:
    - /home/user/.athenas/data/audit.log
  parsers:
    - ndjson:
        target: ""
        add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "athenas-audit"
```
