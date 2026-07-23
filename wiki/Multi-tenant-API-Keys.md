# Multi-tenant API Keys

Athenas Studio supports multi-tenant API keys with per-key quotas, rate limits, and model access control — essential for enterprise deployments where multiple teams or customers share a single inference server.

## Overview

Multi-tenant API keys allow you to:

- **Issue unique API keys** per tenant (team, customer, application)
- **Set per-key rate limits** (requests per minute, tokens per day)
- **Restrict model access** (only allow certain models per key)
- **Track usage** per key for billing and analytics
- **Revoke keys** instantly

## Configuration

API keys are stored in `~/.athenas/data/api_keys.json` and managed via the CLI or API.

### Enable Multi-tenant Keys

In `~/.athenas/config.toml`:

```toml
[server]
# When api_keys_file is set, multi-tenant key auth is enabled
# The global api_key is used as a master/admin key
api_key = "master-admin-key"
```

### API Key Structure

```json
{
  "key_id": "key_abc123",
  "api_key": "sk-ath-xxxxxxxxxxxxxxxxxxxx",
  "name": "Engineering Team",
  "created_at": "2024-01-15T10:00:00Z",
  "expires_at": null,
  "active": true,
  "rate_limit_per_minute": 60,
  "daily_token_limit": 1000000,
  "allowed_models": ["llama-3.1-8b", "qwen2.5-7b"],
  "metadata": {
    "team": "engineering",
    "cost_center": "eng-123"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `key_id` | string | Internal key identifier |
| `api_key` | string | The API key string (sent in Authorization header) |
| `name` | string | Human-readable name |
| `created_at` | ISO 8601 | Creation timestamp |
| `expires_at` | ISO 8601 or null | Expiration (null = never expires) |
| `active` | bool | Whether the key is active |
| `rate_limit_per_minute` | int | Max requests per minute (0 = unlimited) |
| `daily_token_limit` | int | Max tokens per day (0 = unlimited) |
| `allowed_models` | array | Allowed model IDs (empty = all models) |
| `metadata` | object | Custom metadata for your use case |

## CLI Management

### Create a Key

```bash
athenas keys create \
  --name "Engineering Team" \
  --rate-limit 60 \
  --daily-token-limit 1000000 \
  --models "llama-3.1-8b,qwen2.5-7b"
```

### List Keys

```bash
athenas keys list
```

### Show Key Details

```bash
athenas keys info key_abc123
```

### Revoke a Key

```bash
athenas keys revoke key_abc123
```

### Show Usage Stats

```bash
athenas keys usage key_abc123
```

## API Management

API keys can also be managed via REST endpoints (requires master/admin key):

### Create Key

```
POST /v1/keys
Authorization: Bearer <master-admin-key>
```

```json
{
  "name": "Marketing Team",
  "rate_limit_per_minute": 30,
  "daily_token_limit": 500000,
  "allowed_models": ["llama-3.1-8b"],
  "metadata": {"team": "marketing"}
}
```

### List Keys

```
GET /v1/keys
Authorization: Bearer <master-admin-key>
```

### Get Key Info

```
GET /v1/keys/{key_id}
Authorization: Bearer <master-admin-key>
```

### Revoke Key

```
DELETE /v1/keys/{key_id}
Authorization: Bearer <master-admin-key>
```

### Usage Stats

```
GET /v1/keys/{key_id}/usage
Authorization: Bearer <master-admin-key>
```

Response:

```json
{
  "key_id": "key_abc123",
  "today": {
    "requests": 1250,
    "tokens_prompt": 45000,
    "tokens_generated": 32000,
    "tokens_total": 77000
  },
  "rate_limit_remaining": 45,
  "daily_token_remaining": 923000
}
```

## How It Works

1. **Authentication**: Every request must include `Authorization: Bearer <api_key>`. The middleware checks the key against the key store.

2. **Rate Limiting**: Per-key rate limits are enforced in addition to the global IP-based rate limiter. A token bucket algorithm is used per key.

3. **Model Access**: If `allowed_models` is non-empty, requests specifying a model not in the list are rejected with `403 Forbidden`.

4. **Token Quota**: Daily token usage is tracked per key. When the daily limit is exceeded, requests are rejected with `429 Too Many Requests` and a `Retry-After` header.

5. **Usage Tracking**: All requests are logged with the key ID, token counts, and model used. This data is available via the usage API and [Audit Logging](Audit-Logging).

## Usage with OpenAI Client

```python
from openai import OpenAI

# Each tenant uses their own API key
client = OpenAI(
    base_url="http://inference.company.com:8080/v1",
    api_key="sk-ath-xxxxxxxxxxxx"  # tenant-specific key
)

response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Security Best Practices

- **Never commit API keys** to version control
- **Use the master/admin key** only for key management
- **Set expiration dates** for temporary access
- **Restrict models** to prevent unauthorized access to expensive models
- **Monitor usage** regularly for anomalous patterns
- **Rotate keys** periodically
- **Use HTTPS** in production to protect keys in transit
