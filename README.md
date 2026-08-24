# Athenas Studio

> A powerful CLI/TUI tool for running LLM models locally with CUDA, ROCm, and vLLM support. Compatible with HuggingFace model hub and OpenAI API.

[![CI](https://github.com/Kodjaoglanian/athenas-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/Kodjaoglanian/athenas-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org)

---

```
 ░▒▓██████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░ ░▒▓███████▓▒░       ░▒▓███████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓████████▓▒░ ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░░▒▓██████▓▒░        ░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░             ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░             ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░       ░▒▓███████▓▒░   ░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓██████▓▒░  
```

## Features

- **TUI Interface** — Interactive chat with streaming, model selection, real-time stats, and server management
- **Markdown Rendering** — Assistant messages render with markdown formatting (headers, bold, italic, code blocks, lists, blockquotes, links)
- **Multi-line Input** — Write multi-line prompts with Shift+Enter for newlines, Enter to send
- **System Prompt** — Set custom system prompts per conversation via `/system` command
- **Cancel Generation** — Press Esc to cancel ongoing generation and save resources
- **Chat Shortcuts** — PageUp/PageDown/Home/End for fast navigation, Tab to toggle reasoning
- **Semantic Cache** — Server-side response caching based on semantic similarity (cosine similarity + TTL + LRU eviction), reducing token usage and latency for repeated queries
- **Multi-Model Management** — Load, unload, and switch between multiple models simultaneously in the TUI server panel
- **Multimodal Model Support** — Automatic mmproj (multimodal projector) detection and download for vision models (llama.cpp)
- **CLI Commands** — Full command-line interface for scripting and automation
- **Multiple Backends** — llama.cpp (CUDA/ROCm/Vulkan/CPU) and vLLM (CUDA/ROCm)
- **GPU Auto-Download** — Automatically downloads the correct GPU-accelerated llama-server binary (Vulkan for Linux, CUDA for Windows, Metal for macOS) based on detected hardware
- **HuggingFace Integration** — Search, download, and manage models from HuggingFace Hub with automatic mmproj download
- **OpenAI-Compatible API Server** — Drop-in replacement for OpenAI API endpoints with multi-model support
- **Reasoning/Thinking Mode** — Support for reasoning models (Qwen3.5, DeepSeek R1, etc.) with configurable thinking budget
- **Audio Transcription (Whisper)** — Transcribe audio files with Whisper models via CLI (`athenas transcribe`) or API (`/v1/audio/transcriptions`). Auto-downloads whisper-cli binary, supports text/JSON/SRT/VTT output formats, and language hints
- **Hardware Auto-Detection** — Automatically detects CUDA, ROCm, Vulkan, and Metal
- **Auto Resource Limits** — Automatically caps threads, context size, and batch size based on available hardware
- **RAM Pre-Flight Check** — Server panel shows live RAM/VRAM usage and blocks startup if the selected model doesn't fit (with Auto Resource Limits enabled)
- **API Key Generation** — Generate random admin keys with OS entropy directly from the TUI server panel
- **Streaming** — Real-time token streaming in both TUI and API server
- **File Upload** — Upload images and files via `/v1/files` endpoint for multimodal inference
- **Self-Update** — Built-in `athenas update` command to upgrade to the latest release
- **Model Management** — List, search, download, inspect, and remove local models
- **Backend Benchmarking** — Compare backend performance with `athenas backend benchmark`
- **LoRA Adapters** — Load multiple LoRA adapters for model customization
- **Parallel Inference Slots** — Configurable parallel decoding slots for batched inference
- **Vector Store** — Integrated vector store for RAG (retrieval-augmented generation)
- **OpenTelemetry Tracing** — Distributed tracing with OTLP export for observability
- **IP Filtering** — Allowlist/denylist for IP-based access control
- **RemoteBackend** — TUI chat automatically connects to detached server via HTTP API

## Installation

### One-Line Install (Linux & macOS)

```bash
curl -fsSL https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.sh | bash
```

### One-Line Install (Windows PowerShell)

```powershell
irm https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.ps1 | iex
```

### Supported Platforms

| OS | Architecture | Target |
|----|-------------|--------|
| Linux | x86_64 | `x86_64-unknown-linux-gnu` |
| Linux | x86_64 (musl) | `x86_64-unknown-linux-musl` |
| Linux | ARM64 | `aarch64-unknown-linux-gnu` |
| Linux | ARM64 (musl) | `aarch64-unknown-linux-musl` |
| macOS (Intel) | x86_64 | `x86_64-apple-darwin` |
| macOS (Apple Silicon) | ARM64 | `aarch64-apple-darwin` |
| Windows | x86_64 | `x86_64-pc-windows-msvc` |
| Windows | ARM64 | `aarch64-pc-windows-msvc` |

The installer automatically detects your platform, downloads the latest release, verifies the SHA256 checksum, installs the binary to `~/.athenas/bin`, and adds it to your PATH.

### From Source

```bash
git clone https://github.com/Kodjaoglanian/athenas-studio.git
cd athenas-studio
cargo build --release
# Binary at target/release/athenas
```

### Prerequisites

- **Rust** 1.70+ (install via [rustup](https://rustup.rs)) — only needed for building from source
- **llama.cpp** — **Automatically downloaded** on first run. The correct GPU-accelerated binary (Vulkan/CUDA/ROCm/Metal) is selected based on your hardware. You can also install `llama-server` manually in PATH if you prefer a custom build.
- **vLLM** — `pip install vllm` (for vLLM backend, requires CUDA or ROCm)

#### GPU Support on Linux

On Linux, athenas-studio downloads the **Vulkan** binary from llama.cpp releases (no CUDA prebuilt available for Linux). Vulkan works with NVIDIA, AMD, and Intel GPUs.

Make sure you have Vulkan libraries installed:
```bash
# Ubuntu/Debian
sudo apt install -y libvulkan1 mesa-vulkan-drivers

# Fedora
sudo dnf install -y vulkan-loader

# Arch
sudo pacman -S vulkan-icd-loader
```

For NVIDIA GPUs, also ensure your driver supports Vulkan (driver >= 390.x).

### Global Flags

All commands support these optional flags:

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Enable info-level logging |
| `--debug` | `-d` | Enable debug-level logging |

## Usage

### Start TUI (default)
```bash
athenas
```

The TUI provides 6 tabs (F1–F6):

| Key | Tab | Description |
|-----|-----|-------------|
| F1 | Chat | Interactive chat with streaming responses |
| F2 | Models | List local models, load/unload models |
| F3 | Browser | Search and download models from HuggingFace |
| F4 | Server | Configure and manage the API server with multi-model support |
| F5 | Settings | Edit inference settings (GPU, temperature, threads, etc.) |
| F6 | Logs | Live server logs and tracing output |

You can also use **Tab** to cycle between tabs.

#### TUI Settings (F5)

The settings page lets you configure all inference parameters:
- **Device** — Toggle between CPU and GPU (Enter to toggle)
- **GPU Runtime** — Cycle through Auto/CUDA/ROCm/Vulkan/Metal/CPU (Enter to cycle)
- **GPU Layers** — Number of layers to offload to GPU (-1 = all)
- **Temperature, Top-P, Max Tokens** — Sampling parameters
- **Context Size, Batch Size, Threads** — Performance tuning
- **Flash Attention, Streaming, Reasoning** — Feature toggles

All changes are saved to `~/.athenas/config.toml` and applied immediately.

#### TUI Chat (F1)

The chat tab provides an interactive conversation interface with the loaded model:

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| Enter | Send message |
| Shift+Enter | Insert newline (multi-line input) |
| Esc | Cancel ongoing generation |
| Tab | Toggle reasoning/thinking section |
| Up/Down | Scroll chat (single-line input) / navigate input (multi-line) |
| Ctrl+Up/Down | Always scroll chat |
| PageUp/PageDown | Jump 20 lines |
| Home/End | Jump to top/bottom |

**Chat commands:**
| Command | Description |
|---------|-------------|
| `/system <prompt>` | Set a custom system prompt for the model |
| `/system` | Show current system prompt |
| `/system clear` | Remove system prompt |
| `/clear` | Clear all messages |
| `/unload` | Unload the current model from memory |
| `/model` or `/models` | Switch to model list (F2) |
| `/browser` | Switch to HuggingFace browser (F3) |
| `/server` | Switch to server panel (F4) |
| `/settings` | Switch to settings (F5) |
| `/logs` | Switch to logs (F6) |
| `/help` | Show all commands |
| `/quit` | Show quit instructions (Ctrl+C) |

**Markdown rendering:** Assistant messages are rendered with markdown formatting — headers, bold, italic, inline code, code blocks (with language label), lists, blockquotes, horizontal rules, and links. User and system messages display as plain text.

**Reasoning/thinking:** Models that produce reasoning tokens (Qwen3.5, DeepSeek R1) show a collapsible "Thinking" section. Press Tab to expand/collapse. The section shows a preview when collapsed.

#### TUI Server Panel — Multi-Model Management (F4)

When the server is running, you can:
1. Use **Left/Right** on the **Model** field to select a different model
2. Navigate to **▶ Load Additional Model** and press **Enter** to load it alongside the existing model
3. Use **■ Unload** (Left/Right to select, Enter to unload) to remove a model from memory
4. Use **★ Default** (Left/Right to select, Enter to set) to choose which model handles requests without a `model` field
5. The **LOADED MODELS** section shows all active models with their IDs, backends, and default status (★)

#### TUI Server Panel — Enterprise Configuration (F4)

The server panel (F4) also provides full configuration for enterprise features:

- **ADVANCED:** Parallel Slots, LoRA Adapters (comma-separated paths)
- **VECTOR STORE:** Enable/disable, Max Documents, Default Top-K
- **TRACING:** OpenTelemetry enable, OTLP Endpoint, Service Name, Sample Ratio
- **SECURITY:** IP Allowlist, IP Denylist (comma-separated IPs/CIDRs)

The hardware banner shows live **free/total RAM** and **free VRAM**, plus an **Est. load** line with a colored verdict (✓ fits / ⚠ tight / ✗ does NOT fit) that updates as you cycle models — so you know whether a model will fit *before* starting the server.

When Auto Resource Limits is enabled, a **RAM pre-flight check** blocks startup with a clear message if the selected model doesn't fit in available memory.

All fields are editable (Enter to edit, Esc to cancel) or toggleable (Enter to toggle ON/OFF), and persist to `~/.athenas/config.toml`.

**Navigation shortcuts:**
- **PageUp / PageDown** — Jump 10 fields at a time
- **G** (on API Key field) — Generate a random admin key (OS-entropy UUIDv4)
- **X** (on API Key field) — Clear the API key (disables auth)

### Chat in terminal
```bash
athenas chat model.gguf
athenas chat --backend llama.cpp --gpu-layers -1 --context-size 4096
```

### One-shot inference
```bash
athenas run model.gguf "What is the meaning of life?"
athenas run model.gguf "Explain quantum computing" --temperature 0.3 --max-tokens 512
```

### Start API server
```bash
athenas serve model.gguf --port 8080
athenas serve model.gguf --host 0.0.0.0 --port 8080 --backend vllm
```

#### Production server flags

```bash
athenas serve model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --max-concurrent 20 \
  --rate-limit 50 \
  --timeout 300 \
  --max-body-size 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max-concurrent` | 10 | Max simultaneous inference requests (semaphore) |
| `--rate-limit` | 20 | Requests per second per IP (token bucket) |
| `--timeout` | 300 | Request timeout in seconds |
| `--max-body-size` | 10 | Max request body size in MB |

### Search HuggingFace
```bash
athenas models search "llama 3" --gguf
athenas models search "mistral" --pipeline text-generation
```

### Download a model
```bash
athenas models pull TheBloke/Llama-2-7B-Chat-GGUF
athenas models pull TheBloke/Llama-2-7B-Chat-GGUF --file llama-2-7b-chat.Q4_K_M.gguf
```

When pulling a multimodal model (e.g., Llama-3.2-Vision), the mmproj file is automatically detected and downloaded alongside the model. No manual configuration needed — the mmproj is auto-detected at load time.

### List local models
```bash
athenas models list
```

### Show model details
```bash
athenas models info llama-2-7b-chat
```

### Remove a local model
```bash
athenas models remove llama-2-7b-chat
```

### Show hardware info
```bash
athenas hardware
```

### List backends
```bash
athenas backend list
```

### Benchmark backends
```bash
athenas backend benchmark
athenas backend benchmark --model model.gguf
```

### Configuration
```bash
athenas config show
athenas config get inference.default_backend
athenas config set inference.default_backend llama.cpp
athenas config set huggingface.token hf_xxxxx
athenas config init  # reset to defaults
```

### Login to HuggingFace Hub
```bash
athenas login --token hf_xxxxx
```

### Update athenas to latest release
```bash
athenas update
```

### Transcribe audio with Whisper
```bash
# Basic transcription (text output)
athenas transcribe audio.wav --model whisper-large-v3-Q4_K_M.gguf

# With language hint and JSON output
athenas transcribe audio.mp3 --model whisper-large-v3-Q4_K_M.gguf --language pt --format json

# Translate to English with SRT subtitles
athenas transcribe audio.flac --model whisper-large-v3-turbo-Q4_K_M.gguf --translate --format srt
```

Supported audio formats: WAV, MP3, FLAC, OGG, M4A, and more (whisper.cpp handles conversion automatically).

Output formats:
- `text` — Plain text transcription (default)
- `json` — Structured JSON with segments and timestamps
- `srt` — SubRip subtitle format
- `vtt` — WebVTT subtitle format

## API Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (with SSE streaming) |
| `/v1/completions` | POST | Text completions (with SSE streaming) |
| `/v1/models` | GET | List loaded models |
| `/v1/models/load` | POST | Load an additional model at runtime |
| `/v1/models/unload` | POST | Unload a model by ID |
| `/v1/files` | POST | Upload files for multimodal inference (images, documents) |
| `/v1/audio/transcriptions` | POST | Transcribe audio with Whisper (multipart: file, model, language, response_format) |
| `/v1/health` | GET | Health check with model info, uptime, and backend status |
| `/v1/ready` | GET | Kubernetes readiness probe (503 if no model loaded) |
| `/health` | GET | Alias for `/v1/health` |
| `/metrics` | GET | Prometheus-compatible metrics (request count, latency, tokens, errors) |
| `/v1/cache/stats` | GET | Semantic cache statistics (hits, misses, evictions, entries) |
| `/v1/cache/clear` | POST | Clear the semantic cache |

### Multi-Model API

The server supports loading multiple models simultaneously. Each model gets a unique ID.

```bash
# Load an additional model at runtime
curl http://127.0.0.1:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/path/to/second-model.gguf",
    "gpu_layers": -1,
    "context_size": 4096,
    "mmproj_path": "/path/to/mmproj.gguf",
    "set_default": false
  }'

# Unload a model by ID
curl http://127.0.0.1:8080/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model_id": "model-1"}'
```

### Multimodal API

Upload images and use them in chat completions:

```bash
# Upload an image
curl http://127.0.0.1:8080/v1/files \
  -F "file=@photo.jpg"

# Use the image in a chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-vision",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What is in this image?"},
      {"type": "image_url", "image_url": {"url": "file:photo.jpg"}}
    ]}],
    "stream": false
  }'
```

### Example: Using with curl
```bash
# Chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Using with OpenAI Python client
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="llama-2-7b-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Audio transcription via API

```bash
# Transcribe via curl (multipart form)
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v3-Q4_K_M.gguf" \
  -F "language=pt" \
  -F "response_format=json"

# Get SRT subtitles
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3-turbo-Q4_K_M.gguf" \
  -F "response_format=srt"
```

## Architecture

```
athenas-studio/
├── crates/
│   ├── athenas-core/        # Config, storage, hardware detection, model registry
│   ├── athenas-inference/   # Backend trait, llama.cpp, vLLM & RemoteBackend implementations
│   ├── athenas-hub/         # HuggingFace API client, download manager
│   ├── athenas-server/      # OpenAI-compatible API server (axum), multi-model manager
│   ├── athenas-tui/         # Terminal UI (ratatui + crossterm), server panel with multi-model, enterprise configs
│   └── athenas-cli/         # CLI entry point (clap)
├── .github/workflows/       # CI, release & PR build pipelines
├── install.sh               # Linux/macOS installer script
├── install.ps1              # Windows installer script
├── Cargo.toml               # Workspace
├── LICENSE                  # MIT
├── CONTRIBUTING.md          # Contribution guide
└── README.md                # This file
```

## Configuration

Config file: `~/.athenas/config.toml`

Models directory: `~/.athenas/models/`

```toml
version = "0.9.7"

[paths]
models_dir = "~/.athenas/models"
cache_dir = "~/.athenas/cache"
data_dir = "~/.athenas/data"

[inference]
default_backend = "auto"        # auto, llama.cpp, vllm
default_gpu_layers = -1         # -1 = all, 0 = CPU only
gpu_runtime = "auto"            # auto, cuda, rocm, vulkan, metal, cpu
# gpu_device = 0                # GPU index (0, 1, 2, ...). None = auto
default_context_size = 4096
default_batch_size = 512
default_threads = 0             # 0 = auto-detect (leaves 1 core free)
flash_attention = true
default_temperature = 0.7
default_top_p = 0.9
default_max_tokens = 2048
streaming_enabled = true
# Reasoning/Thinking mode (Qwen3.5, DeepSeek R1, etc.)
reasoning_enabled = true
reasoning_budget = -1           # -1 = unlimited, 0 = off, N = token limit
# Hardware protection
ram_reserve_mb = 2048           # MB reserved for OS
cpu_reserve_cores = 1           # cores to leave free
auto_resource_limits = true     # auto-cap threads/ctx/batch based on hardware
# Advanced inference
lora_paths = []                 # LoRA adapter paths (e.g. ["/path/to/adapter.gguf"])
parallel_slots = 4              # parallel decoding slots (1=safe, 4=resilient but more RAM)

[server]
default_host = "127.0.0.1"
default_port = 8080
cors_enabled = true
# api_key = "your-secret-key"   # optional auth
max_concurrent_requests = 10    # max simultaneous inferences
rate_limit_per_second = 20      # token bucket per IP
request_timeout_secs = 300      # kill stuck requests (must be >= prompt processing time)
max_body_size_mb = 10           # DoS protection
enable_metrics = true           # Prometheus /metrics endpoint
enable_compression = true       # gzip response compression
# IP filtering (empty allowlist = allow all)
ip_allowlist = []               # e.g. ["10.0.0.0/8", "192.168.1.100"]
ip_denylist = []                # e.g. ["10.0.0.5"]

[server.vector_store]
enabled = false                 # enable integrated vector store for RAG
max_documents = 0               # 0 = unlimited
default_top_k = 5               # default search results count

[server.otel]
enabled = false                 # enable OpenTelemetry distributed tracing
# endpoint = "http://localhost:4317"  # OTLP endpoint
service_name = "athenas-studio" # service name for traces
sample_ratio = 1.0              # sampling ratio 0.0-1.0

[server.semantic_cache]
enabled = false                 # enable semantic caching for chat completions
similarity_threshold = 0.85     # cosine similarity threshold (0.0-1.0, higher = stricter matching)
ttl_secs = 3600                 # cache entry time-to-live in seconds (1 hour default)
max_entries = 1000              # maximum cache entries (LRU eviction when exceeded)

[huggingface]
# token = "hf_xxxxx"            # for gated models
default_revision = "main"

[logging]
level = "info"                  # trace, debug, info, warn, error
file_logging = false
```

## Backends

### llama.cpp
- **Best for:** Single-user inference, GGUF models, CPU/GPU mix, multimodal models
- **GPU support:** CUDA, ROCm, Vulkan, Metal
- **Multimodal:** Automatically detects and uses mmproj files for vision models
- **Install:** **Automatic** — the correct GPU-accelerated `llama-server` binary is downloaded on first run based on your OS and GPU. You can also build [llama.cpp](https://github.com/ggml-org/llama.cpp) manually and add `llama-server` to PATH.

| Platform | GPU | Binary downloaded |
|----------|-----|--------------------|
| Linux + NVIDIA | Yes | `bin-ubuntu-vulkan-x64.tar.gz` |
| Linux + AMD | Yes | `bin-ubuntu-vulkan-x64.tar.gz` |
| Linux + Intel | Yes | `bin-ubuntu-vulkan-x64.tar.gz` |
| Linux | None | `bin-ubuntu-x64.tar.gz` (CPU-only) |
| Windows + NVIDIA | Yes | `bin-win-cuda-12.4-x64.zip` |
| Windows + AMD | Yes | `bin-win-vulkan-x64.zip` (ROCm fallback) |
| macOS | Apple Silicon | Metal (built into all macOS binaries) |

**Note:** On Linux, there is no CUDA prebuilt binary from llama.cpp. We use Vulkan instead, which works with all GPU vendors. Performance is comparable to CUDA for inference.

### vLLM
- **Best for:** High-throughput serving, multi-user, PagedAttention
- **GPU support:** CUDA, ROCm
- **Install:** `pip install vllm`

## CI/CD

The project includes three GitHub Actions workflows:

- **CI** (`ci.yml`) — Formatting checks, clippy lints, tests, and cross-compilation for all supported targets
- **Release** (`release.yml`) — Triggered on version tags (`v*`), builds and publishes binaries for all platforms with SHA256 checksums and install scripts
- **PR Build** (`pr-build.yml`) — Build verification on pull requests

## License

MIT — See [LICENSE](LICENSE)
