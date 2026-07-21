# Athenas Studio

> A powerful CLI/TUI tool for running LLM models locally with CUDA, ROCm, and vLLM support. Compatible with HuggingFace model hub and OpenAI API.

[![CI](https://github.com/Kodjaoglanian/athenas-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/Kodjaoglanian/athenas-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org)

---

```
    ╔══════════════════════════════════════════════════╗
    ║                                                  ║
    ║    █████╗ ███████╗██╗   ██╗███████╗███████╗      ║
    ║   ██╔══██╗╚══███╔╝██║   ██║██╔════╝██╔════╝      ║
    ║   ███████║  ███╔╝ ██║   ██║███████╗███████╗      ║
    ║   ██╔══██║ ███╔╝  ██║   ██║╚════██║╚════██║      ║
    ║   ██║  ██║███████╗╚██████╔╝███████║███████║      ║
    ║   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚══════╝      ║
    ║                                                  ║
    ║          S T U D I O   ·   L L M   T O O L       ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
```

## Features

- **TUI Interface** — Interactive chat with streaming, model selection, and real-time stats
- **CLI Commands** — Full command-line interface for scripting and automation
- **Multiple Backends** — llama.cpp (CUDA/ROCm/Vulkan/CPU) and vLLM (CUDA/ROCm)
- **HuggingFace Integration** — Search, download, and manage models from HuggingFace Hub
- **OpenAI-Compatible API Server** — Drop-in replacement for OpenAI API endpoints
- **Hardware Auto-Detection** — Automatically detects CUDA, ROCm, Vulkan, and Metal
- **Streaming** — Real-time token streaming in both TUI and API server

## Installation

### One-Line Install (Linux & macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/Kodjaoglanian/athenas-studio/main/install.sh | bash
```

### One-Line Install (Windows PowerShell)

```powershell
irm https://raw.githubusercontent.com/Kodjaoglanian/athenas-studio/main/install.ps1 | iex
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
```

### Start API server
```bash
athenas serve model.gguf --port 8080
```

### Search HuggingFace
```bash
athenas models search "llama 3" --gguf
```

### Download a model
```bash
athenas models pull TheBloke/Llama-2-7B-Chat-GGUF
```

### List local models
```bash
athenas models list
```

### Show hardware info
```bash
athenas hardware
```

### List backends
```bash
athenas backend list
```

### Configuration
```bash
athenas config show
athenas config set inference.default_backend llama.cpp
athenas config set huggingface.token hf_xxxxx
```

## API Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (with SSE streaming) |
| `/v1/completions` | POST | Text completions (with SSE streaming) |
| `/v1/models` | GET | List loaded models |
| `/v1/health` | GET | Health check |

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
├── .github/workflows/       # CI pipeline
├── Cargo.toml               # Workspace
├── LICENSE                  # MIT
├── CONTRIBUTING.md          # Contribution guide
└── README.md                # This file
```

## Configuration

Config file: `~/.athenas/config.toml`

Models directory: `~/.athenas/models/`

```toml
version = "0.1.0"

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

## License

MIT — See [LICENSE](LICENSE)
