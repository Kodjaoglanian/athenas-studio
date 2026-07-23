# Configuration

Config file: `~/.athenas/config.toml`

Models directory: `~/.athenas/models/`
Cache directory: `~/.athenas/cache/`
Data directory: `~/.athenas/data/`

## Full Config Reference

```toml
version = "0.4.0"

[paths]
models_dir = "~/.athenas/models"
cache_dir = "~/.athenas/cache"
data_dir = "~/.athenas/data"

[inference]
default_backend = "auto"           # auto, llama.cpp, vllm
default_gpu_layers = -1            # -1 = all layers on GPU
default_context_size = 4096        # context window size
default_batch_size = 512           # prompt processing batch size
default_threads = 0                # 0 = auto-detect (leaves 1 core free)
flash_attention = true             # flash attention for faster inference
default_temperature = 0.7
default_top_p = 0.9
default_max_tokens = 2048
streaming_enabled = true
# Reasoning/Thinking mode (Qwen3.5, DeepSeek R1, etc.)
reasoning_enabled = true
reasoning_budget = -1              # -1 = unlimited, 0 = off, N = token limit
# Hardware protection
ram_reserve_mb = 2048              # MB reserved for OS
cpu_reserve_cores = 1              # cores to leave free
auto_resource_limits = true        # auto-cap threads/ctx/batch based on hardware

[server]
default_host = "127.0.0.1"
default_port = 8080
cors_enabled = true
# api_key = "your-secret-key"      # optional global auth (use multi-tenant keys for enterprise)
max_concurrent_requests = 10       # max simultaneous inferences (semaphore)
rate_limit_per_second = 20         # token bucket per IP
request_timeout_secs = 120         # kill stuck requests
max_body_size_mb = 10              # DoS protection
enable_metrics = true              # Prometheus /metrics endpoint
enable_compression = true          # gzip response compression

[huggingface]
# token = "hf_xxxxx"               # for gated models
default_revision = "main"

[logging]
level = "info"                     # trace, debug, info, warn, error
file_logging = false
```

## Managing Configuration

```bash
# Show full config
athenas config show

# Get a specific value
athenas config get inference.default_backend

# Set a value
athenas config set inference.default_backend llama.cpp
athenas config set huggingface.token hf_xxxxx

# Reset to defaults
athenas config init
```

## Environment Variables

The config file is the primary configuration method. However, logging can be controlled via:

- `RUST_LOG` — Override log level (e.g., `RUST_LOG=debug`)
- `ATHENAS_CONFIG` — Override config file path (defaults to `~/.athenas/config.toml`)
