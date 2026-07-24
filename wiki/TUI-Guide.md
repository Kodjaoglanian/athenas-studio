# TUI Guide

The Athenas Studio TUI provides a full-featured terminal interface for LLM inference, model management, and server control.

## Key Bindings

### Global

| Key | Action |
|-----|--------|
| **F1** | Chat panel |
| **F2** | Models panel |
| **F3** | Model browser (HuggingFace) |
| **F4** | Settings |
| **F5** | Server panel |
| **F6** | Logs |
| **Ctrl+C** | Quit |

### Chat Panel (F1)

| Key | Action |
|-----|--------|
| **Enter** | Send message |
| **Shift+Enter** | Newline |
| **Up/Down** | Scroll chat history |
| **Tab** | Switch between input and chat |
| **Esc** | Cancel generation |
| `/help` | Show chat commands |
| `/clear` | Clear chat history |
| `/model` | Show current model info |

### Models Panel (F2)

| Key | Action |
|-----|--------|
| **Up/Down** | Navigate model list |
| **Enter** | Load selected model |
| **Left/Right** | Cycle models |
| **Delete** | Unload model |

### Model Browser (F3)

| Key | Action |
|-----|--------|
| **Enter** | Search / Download selected model |
| **Up/Down** | Navigate results |
| **Esc** | Clear search |

### Settings (F4)

| Key | Action |
|-----|--------|
| **Up/Down** | Navigate settings |
| **Left/Right** | Change value |
| **Enter** | Edit text field |
| **Esc** | Cancel edit |

### Server Panel (F5)

| Key | Action |
|-----|--------|
| **Up/Down** | Navigate fields |
| **Left/Right** | Change values / select models |
| **Enter** | Start server / Load additional model / Unload / Set default |
| **Esc** | Stop server |

### Logs (F6)

| Key | Action |
|-----|--------|
| **Up/Down** | Scroll logs |
| **C** | Clear logs |
| **Auto-scroll** | Follows latest log entries |

## Server Panel — Multi-Model Management

When the server is running, you can:

1. Use **Left/Right** on the **Model** field to select a different model
2. Navigate to **▶ Load Additional Model** and press **Enter** to load it alongside the existing model
3. Use **■ Unload** (Left/Right to select, Enter to unload) to remove a model from memory
4. Use **★ Default** (Left/Right to select, Enter to set) to choose which model handles requests without a `model` field
5. The **LOADED MODELS** section shows all active models with their IDs, backends, and default status (★)

## Server Panel — Enterprise Configuration

The server panel (F5) provides full configuration for enterprise features, organized in sections:

### MODEL
- **Model** — Select local model (Left/Right to cycle)

### SERVER
- **Host**, **Port**, **API Key**, **Max Concurrent**, **Rate Limit**, **Timeout**, **Max Body Size**, **CORS**, **Metrics**, **Compression**

### OPTIMIZATION
- **Backend**, **GPU Layers**, **Context Size**, **Batch Size**, **Threads**, **Flash Attention**, **Max Tokens**, **Temperature**, **Top P**, **Reasoning**, **Reasoning Budget**, **RAM Reserve**, **CPU Reserve**, **Auto Resource Limits**

### ADVANCED
- **Parallel Slots** — Parallel decoding slots for batched inference (1=safe, 4=fast)
- **LoRA Adapters** — Comma-separated paths to `.gguf` LoRA adapter files

### VECTOR STORE
- **Vector Store** — Toggle ON/OFF to enable integrated RAG vector store
- **VS Max Documents** — Max documents to store (0 = unlimited)
- **VS Default Top-K** — Number of results to retrieve by default

### TRACING
- **OpenTelemetry** — Toggle ON/OFF to enable distributed tracing
- **OTLP Endpoint** — OTLP gRPC endpoint (e.g. `http://localhost:4317`)
- **OTel Service Name** — Service name for traces
- **OTel Sample Ratio** — Sampling ratio 0.0-1.0

### SECURITY
- **IP Allowlist** — Comma-separated IPs/CIDRs (empty = allow all)
- **IP Denylist** — Comma-separated IPs/CIDRs to block

### ACTION
- **Start Server** — Loads model and starts the API server
- **Stop Server** — Stops the running server (kills entire process group)
- **Load Additional Model** — Load another model while server is running
- **Unload Model** — Remove a model from memory
- **Set Default Model** — Choose which model handles requests without a `model` field

All text fields are edited with **Enter** (type value, Enter to save, Esc to cancel). Toggle fields use **Enter** to switch ON/OFF.

## Chat Integration with Server

When the server is running with a loaded model, the TUI chat automatically uses the server's loaded model. This means:

- **No duplicate loading** — The chat reuses the server's model backend
- **Shared context** — Chat and API requests share the same model instance
- **Real-time sync** — Loading/unloading models in the server panel updates the chat state

### Detached Server (RemoteBackend)

When the server is started as a **detached process** from the TUI (F5 → Start Server), the chat uses a `RemoteBackend` that connects to the server via its HTTP API:

- The TUI chat sends requests to `http://{host}:{port}/v1/chat/completions`
- Streaming responses are proxied via SSE
- The chat displays `Using remote server model '...'` when connected
- API key authentication is passed through if configured
- Stopping the server kills the entire process group (athenas serve + llama-server child)

If you also load a model locally via F2, the local model takes priority for chat. The server model (in-process or remote) is used as a fallback when no local model is loaded.

## Reasoning/Thinking Display

For models that support reasoning (Qwen3.5, DeepSeek R1, etc.), the TUI shows:

- **Reasoning content** in a collapsible section above the response
- **Toggle expand/collapse** with Enter on the reasoning section
- **Token-per-second** counter during generation
- If the model produces only reasoning and no response, a helpful message is shown
