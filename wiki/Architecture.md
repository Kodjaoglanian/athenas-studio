# Architecture

Athenas Studio is built as a Rust workspace with modular crates for clean separation of concerns.

## Crate Structure

```
athenas-studio/
├── crates/
│   ├── athenas-core/        # Config, storage, hardware detection, model registry
│   ├── athenas-inference/   # Backend trait, llama.cpp & vLLM implementations
│   ├── athenas-hub/         # HuggingFace API client, download manager
│   ├── athenas-server/      # OpenAI-compatible API server (axum), multi-model manager
│   ├── athenas-tui/         # Terminal UI (ratatui + crossterm)
│   └── athenas-cli/         # CLI entry point (clap)
├── .github/workflows/       # CI, release & PR build pipelines
├── install.sh               # Linux/macOS installer script
├── install.ps1              # Windows installer script
├── Cargo.toml               # Workspace root
├── LICENSE                  # MIT
├── CONTRIBUTING.md          # Contribution guide
└── README.md                # Project README
```

## Crate Dependencies

```
athenas-cli
  ├── athenas-core
  ├── athenas-inference
  ├── athenas-hub
  ├── athenas-server
  └── athenas-tui

athenas-tui
  ├── athenas-core
  ├── athenas-inference
  └── athenas-server

athenas-server
  ├── athenas-core
  └── athenas-inference

athenas-inference
  └── athenas-core

athenas-hub
  └── athenas-core

athenas-core
  (no internal deps)
```

## athenas-core

**Purpose**: Foundation crate with configuration, error types, hardware detection, and model registry.

### Key Types

- `AppConfig` — Root config struct with `InferenceConfig`, `ServerConfig`, `LoggingConfig`
- `BackendType` — Enum: `Auto`, `LlamaCpp`, `Vllm`
- `HardwareInfo` / `HardwareDetector` — CPU, GPU, memory detection
- `ModelRegistry` — Local model registry (SQLite-backed)
- `AthenasError` — Unified error type for the entire workspace

### Config Structure

```rust
pub struct AppConfig {
    pub version: String,
    pub paths: PathsConfig,
    pub inference: InferenceConfig,
    pub server: ServerConfig,
    pub huggingface: HuggingFaceConfig,
    pub logging: LoggingConfig,
}
```

## athenas-inference

**Purpose**: Backend abstraction and implementations for LLM inference.

### Backend Trait

```rust
#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn is_loaded(&self) -> bool;
    async fn load_model(&mut self, config: ModelLoadConfig) -> Result<()>;
    async fn unload_model(&mut self) -> Result<()>;
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;
    async fn chat_stream(&self, request: ChatRequest, tx: mpsc::Sender<StreamChunk>) -> Result<()>;
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    async fn complete_stream(&self, request: CompletionRequest, tx: mpsc::Sender<StreamChunk>) -> Result<()>;
    fn model_info(&self) -> Option<ModelInfo>;
    fn boxed_clone(&self) -> Box<dyn Backend>;
}
```

### Implementations

- `LlamaCppBackend` — Runs `llama-server` subprocess, communicates via HTTP
- `BackendFactory` — Creates backends based on `BackendType` and hardware

### Key Types

- `ChatRequest` / `ChatResponse` — Chat completion types
- `CompletionRequest` / `CompletionResponse` — Text completion types
- `StreamChunk` — Streaming token with reasoning support
- `ModelLoadConfig` — Model loading parameters
- `ChatMessage` / `Role` / `MessageContent` — Message types

## athenas-server

**Purpose**: OpenAI-compatible HTTP API server built with Axum.

### Components

- `ApiServer` — Server initialization and startup
- `ModelManager` — Multi-model management (load, unload, route)
- `SessionManager` — Conversation sessions with slot assignment
- `SlotManager` — KV cache slot management (save/restore/erase)
- `RateLimiter` — Token bucket rate limiting middleware
- `Metrics` — Prometheus metrics collection

### Request Flow

```
Client Request
    │
    ▼
┌───────────────────┐
│ Rate Limiter      │  (per-IP token bucket)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Auth Middleware   │  (API key check)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Metrics Middleware│  (record request metrics)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Route Handler     │  (chat_completions, completions, etc.)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Session Manager   │  (build full message list from history)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Model Manager     │  (find backend by model ID)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Backend (trait)   │  (llama.cpp or vLLM)
└───────┬───────────┘
        │
        ▼
    Response (JSON or SSE)
```

### API Routes

| Route | Handler |
|-------|---------|
| `POST /v1/chat/completions` | `chat_completions` |
| `POST /v1/completions` | `completions` |
| `GET /v1/models` | `list_models` |
| `POST /v1/models/load` | `load_model` |
| `POST /v1/models/unload` | `unload_model` |
| `POST /v1/files` | `upload_file` |
| `GET /v1/health` | `health` |
| `GET /v1/ready` | `ready` |
| `GET /metrics` | `metrics` |
| `POST /v1/sessions` | `create_session` |
| `GET /v1/sessions` | `list_sessions` |
| `GET /v1/sessions/{id}` | `get_session` |
| `DELETE /v1/sessions/{id}` | `delete_session` |
| `GET /v1/slots` | `list_slots` |
| `POST /v1/slots/{id}` | `control_slot` |
| `POST /v1/slots/{id}/save` | `save_slot` |
| `POST /v1/slots/{id}/restore` | `restore_slot` |

## athenas-tui

**Purpose**: Terminal user interface built with Ratatui + Crossterm.

### Components

- `TuiApp` — Main application state and event loop
- `ChatState` — Chat panel state (messages, streaming, model info)
- `ServerPanel` — Server configuration and multi-model management
- `ModelBrowser` — HuggingFace model search and download
- `Settings` — Configuration editor
- `LogBuffer` / `LogBufferLayer` — In-app log viewing (tracing layer)

### TUI Pages

| Key | Page | Description |
|-----|------|-------------|
| F1 | Chat | Interactive chat with streaming |
| F2 | Models | Local model list and loading |
| F3 | Browser | HuggingFace model search/download |
| F4 | Settings | Configuration editor |
| F5 | Server | API server management |
| F6 | Logs | Real-time application logs |

## athenas-hub

**Purpose**: HuggingFace Hub integration for model search and download.

### Features

- Model search with filters (pipeline tag, GGUF only)
- Model download with progress bar
- Automatic mmproj detection and download for multimodal models
- HuggingFace token authentication for gated models
- Resume support for interrupted downloads

## athenas-cli

**Purpose**: CLI entry point using Clap.

### Commands

- `athenas` (TUI) — Default
- `athenas chat` — Interactive chat
- `athenas serve` — API server
- `athenas run` — One-shot inference
- `athenas models` — Model management
- `athenas config` — Configuration
- `athenas hardware` — Hardware info
- `athenas backend` — Backend management
- `athenas login` — HuggingFace login
- `athenas update` — Self-update

## Data Flow: Chat Completion

```
1. Client sends POST /v1/chat/completions
2. Rate limiter checks IP token bucket
3. Auth middleware validates API key
4. Metrics middleware starts timing
5. Semaphore acquired (concurrency control)
6. If session_id provided:
   a. SessionManager builds full message list (history + new)
   b. Slot assignment checked
7. ModelManager finds backend by model ID (or default)
8. Backend.chat() or Backend.chat_stream() called
9. For llama.cpp: HTTP request to llama-server /v1/chat/completions
10. Response parsed (content + reasoning_content + usage)
11. If session_id: response appended to session
12. Metrics recorded (tokens, duration, status)
13. Response returned (JSON or SSE stream)
```

## Build System

- **Workspace**: Single `Cargo.toml` with 6 member crates
- **Release profile**: LTO, single codegen unit, stripped binaries
- **CI**: GitHub Actions with formatting, clippy, tests, cross-compilation
- **Release**: Automated binary builds for 8 platform targets with SHA256 checksums
- **Installers**: `install.sh` (Linux/macOS) and `install.ps1` (Windows)
