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

- **TUI Interface** — Interactive chat with streaming, model selection, and real-time stats
- **CLI Commands** — Full command-line interface for scripting and automation
- **Multiple Backends** — llama.cpp (CUDA/ROCm/Vulkan/CPU) and vLLM (CUDA/ROCm)
- **HuggingFace Integration** — Search, download, and manage models from HuggingFace Hub
- **OpenAI-Compatible API Server** — Drop-in replacement for OpenAI API endpoints
- **Hardware Auto-Detection** — Automatically detects CUDA, ROCm, Vulkan, and Metal
- **Streaming** — Real-time token streaming in both TUI and API server
- **Self-Update** — Built-in `athenas update` command to upgrade to the latest release
- **Model Management** — List, search, download, inspect, and remove local models
- **Backend Benchmarking** — Compare backend performance with `athenas backend benchmark`

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
| Linux | x86_64 | `x86_64-unknown-linux-musl` |
| Linux | ARM64 | `aarch64-unknown-linux-musl` |
| macOS (Intel) | x86_64 | `x86_64-apple-darwin` |
| macOS (Apple Silicon) | ARM64 | `aarch64-apple-darwin` |
| Windows | x86_64 | `x86_64-pc-windows-msvc` |
| Windows | ARM64 | `aarch64-pc-windows-msvc` |

The installer automatically detects your platform, downloads the latest release, verifies the SHA256 checksum, installs the binary to `~/.athenas/bin`, and adds it to your PATH.

### From Source

```bash
git clone https://github.com/Kodjaoglanian/athenas-studio.git
cd athenas
cargo build --release
# Binary at target/release/athenas
```

### Prerequisites

- **Rust** 1.70+ (install via [rustup](https://rustup.rs))
- **llama.cpp** — Install and ensure `llama-server` is in PATH (for llama.cpp backend)
- **vLLM** — `pip install vllm` (for vLLM backend, requires CUDA or ROCm)

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
| `--timeout` | 120 | Request timeout in seconds |
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

## API Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (with SSE streaming) |
| `/v1/completions` | POST | Text completions (with SSE streaming) |
| `/v1/models` | GET | List loaded models |
| `/v1/health` | GET | Health check with model info, uptime, and backend status |
| `/v1/ready` | GET | Kubernetes readiness probe (503 if no model loaded) |
| `/metrics` | GET | Prometheus-compatible metrics (request count, latency, tokens, errors) |

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

## Architecture

```
athenas-studio/
├── crates/
│   ├── athenas-core/        # Config, storage, hardware detection, model registry
│   ├── athenas-inference/   # Backend trait, llama.cpp & vLLM implementations
│   ├── athenas-hub/         # HuggingFace API client, download manager
│   ├── athenas-server/      # OpenAI-compatible API server (axum)
│   ├── athenas-tui/         # Terminal UI (ratatui + crossterm)
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
version = "0.1.4"

[paths]
models_dir = "~/.athenas/models"
cache_dir = "~/.athenas/cache"
data_dir = "~/.athenas/data"

[inference]
default_backend = "auto"        # auto, llama.cpp, vllm
default_gpu_layers = -1         # -1 = all
default_context_size = 4096
default_batch_size = 512
flash_attention = true
default_temperature = 0.7
default_top_p = 0.9
default_max_tokens = 2048

[server]
default_host = "127.0.0.1"
default_port = 8080
cors_enabled = true
# api_key = "your-secret-key"   # optional auth
max_concurrent_requests = 10    # max simultaneous inferences
rate_limit_per_second = 20      # token bucket per IP
request_timeout_secs = 120      # kill stuck requests
max_body_size_mb = 10           # DoS protection
enable_metrics = true           # Prometheus /metrics endpoint
enable_compression = true       # gzip response compression

[huggingface]
# token = "hf_xxxxx"            # for gated models
default_revision = "main"

[logging]
level = "info"                  # trace, debug, info, warn, error
file_logging = false
```

## Backends

### llama.cpp
- **Best for:** Single-user inference, GGUF models, CPU/GPU mix
- **GPU support:** CUDA, ROCm, Vulkan, Metal
- **Install:** Build [llama.cpp](https://github.com/ggerganov/llama.cpp) and add `llama-server` to PATH

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
