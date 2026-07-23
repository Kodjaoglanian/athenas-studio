# Session & Slot Management

Athenas Studio provides server-side session management with KV cache persistence for efficient multi-turn conversations.

## Sessions

Sessions maintain conversation history on the server, so clients don't need to resend the full context with each request.

### Creating a Session

```
POST /v1/sessions
```

```json
{
  "session_id": "my-conversation",
  "system_prompt": "You are a helpful assistant."
}
```

If `session_id` is omitted, a random ID is generated. The response includes the session ID:

```json
{
  "session_id": "sess_abc123def456",
  "slot_id": 0,
  "created": true
}
```

### Using a Session

Include the `session_id` in chat completion requests:

```json
{
  "model": "llama-3.1-8b",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "session_id": "sess_abc123def456"
}
```

The server automatically:
1. Prepends the system prompt (if set)
2. Includes all previous messages from the session
3. Appends the new message
4. Sends the full context to the model
5. Appends the response to the session history

### Listing Sessions

```
GET /v1/sessions
```

```json
{
  "sessions": [
    {
      "id": "sess_abc123",
      "model_id": "llama-3.1-8b",
      "message_count": 12,
      "slot_id": 0,
      "slot_cache_warm": true,
      "created_at_secs": 3600,
      "last_active_secs": 120,
      "system_prompt": "You are a helpful assistant."
    }
  ]
}
```

### Getting Session Details

```
GET /v1/sessions/{id}
```

### Deleting a Session

```
DELETE /v1/sessions/{id}
```

This also erases the associated slot's KV cache.

### Session Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Max history | 100 messages | Messages are trimmed (oldest first) when exceeded |
| TTL | 2 hours | Inactive sessions are automatically purged |
| Max slots | 4 | Number of parallel slots (set by `--parallel` flag) |

## Slot Management

The llama-server uses "slots" for parallel inference. Each slot maintains its own KV cache, allowing multiple conversations to run simultaneously without reprocessing context.

### How Slots Work

1. When a session is created, it's assigned a slot (round-robin)
2. On subsequent requests, the slot's KV cache already contains the previous context
3. Only new tokens need to be processed — dramatically reducing latency
4. The `slot_cache_warm` flag indicates whether the cache contains prior context

### Listing Slots

```
GET /v1/slots
```

```json
[
  {
    "id": 0,
    "n_ctx": 4096,
    "n_past": 150,
    "n_tokens": 150,
    "is_processing": false,
    "prompt": "You are a helpful assistant...",
    "cache": {
      "tokens": 150,
      "used_tokens": 150,
      "parent": null
    }
  }
]
```

### Saving a Slot Checkpoint

Save the KV cache state to disk for later restoration:

```
POST /v1/slots/{id}/save
```

```json
{"name": "conversation-checkpoint-1"}
```

This is useful for:
- Pausing long conversations and resuming later
- Creating reusable context templates
- Saving expensive prompt processing (e.g., large documents)

### Restoring a Slot Checkpoint

```
POST /v1/slots/{id}/restore
```

```json
{"name": "conversation-checkpoint-1"}
```

The slot's KV cache is restored from the checkpoint, avoiding the need to reprocess the full context.

### Erasing a Slot

Free the KV cache memory:

```
POST /v1/slots/{id}
```

```json
{"erase": true}
```

### Slot Assignment

Slots are assigned automatically when sessions are created. The assignment is tracked internally:

```
GET /v1/slots/{id}/assignment
```

```json
{
  "slot_id": 0,
  "session_id": "sess_abc123"
}
```

## KV Cache Persistence Flow

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐
│  Request 1  │────▶│  Slot 0  │────▶│ KV Cache:   │
│  "Hello"    │     │  Active  │     │ [Hello]     │
└─────────────┘     └──────────┘     └─────────────┘
                                           │
┌─────────────┐     ┌──────────┐          │ (warm cache)
│  Request 2  │────▶│  Slot 0  │──────────┘
│  "How are   │     │  Active  │────▶│ KV Cache:     │
│   you?"     │     └──────────┘     │ [Hello, How   │
└─────────────┘                      │  are you?]    │
                                      └───────────────┘
```

- **Request 1**: Full context is processed. KV cache is populated.
- **Request 2**: Only new tokens are processed. Previous KV cache is reused.
- **Save**: KV cache is written to disk as a checkpoint.
- **Restore**: Checkpoint is loaded back into the slot.
- **Erase**: KV cache is cleared, freeing memory.

## Best Practices

- **Use sessions for multi-turn conversations** — Avoid resending full context each time
- **Save checkpoints for long-running tasks** — Protect against crashes
- **Erase slots when done** — Free GPU/CPU memory
- **Monitor slot usage** — Use `GET /v1/slots` to see cache state
- **Invalidate caches after model reload** — The server does this automatically
