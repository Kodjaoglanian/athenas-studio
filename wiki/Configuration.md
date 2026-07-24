# Configuration

Config file: `~/.athenas/config.toml`

Models directory: `~/.athenas/models/`
Cache directory: `~/.athenas/cache/`
Data directory: `~/.athenas/data/`

## Full Config Reference

```toml
version = "0.7.4"

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
# Advanced inference
lora_paths = []                    # LoRA adapter paths (e.g. ["/path/to/adapter.gguf"])
parallel_slots = 1                 # parallel decoding slots (1=safe, 4=fast but more RAM)

[server]
default_host = "127.0.0.1"
default_port = 8080
cors_enabled = true
# api_key = "your-secret-key"      # optional global auth
max_concurrent_requests = 10       # max simultaneous inferences (semaphore)
rate_limit_per_second = 20         # token bucket per IP
request_timeout_secs = 120         # kill stuck requests
max_body_size_mb = 10              # DoS protection
enable_metrics = true              # Prometheus /metrics endpoint
enable_compression = true          # gzip response compression
# IP filtering (empty allowlist = allow all)
ip_allowlist = []                  # e.g. ["10.0.0.0/8", "192.168.1.100"]
ip_denylist = []                   # e.g. ["10.0.0.5"]

[server.vector_store]
enabled = false                    # enable integrated vector store for RAG
max_documents = 0                  # 0 = unlimited
default_top_k = 5                  # default search results count

[server.otel]
enabled = false                    # enable OpenTelemetry distributed tracing
# endpoint = "http://localhost:4317"  # OTLP endpoint
service_name = "athenas-studio"    # service name for traces
sample_ratio = 1.0                 # sampling ratio 0.0-1.0

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

## Advanced Inference

### LoRA Adapters

LoRA (Low-Rank Adaptation) adapters allow fine-tuning models without modifying the base weights. Configure multiple adapters:

```toml
[inference]
lora_paths = ["/path/to/adapter1.gguf", "/path/to/adapter2.gguf"]
```

Or via TUI Server Panel (F5) → **LoRA Adapters** field (comma-separated paths).

### Parallel Inference Slots

Parallel decoding slots enable batched inference for higher throughput in multi-user scenarios:

```toml
[inference]
parallel_slots = 1    # 1 = safe (default), 4 = fast but uses more RAM
```

- **1 slot** — Default, lowest memory usage, best for single-user
- **4 slots** — Higher throughput for concurrent API requests, requires more GPU/CPU memory

## Vector Store

The integrated vector store enables RAG (retrieval-augmented generation) by storing and retrieving document embeddings:

```toml
[server.vector_store]
enabled = false                    # set to true to enable
max_documents = 0                  # 0 = unlimited, N = limit
default_top_k = 5                  # number of results to retrieve
```

## OpenTelemetry Tracing

Distributed tracing with OTLP export for observability and performance analysis:

```toml
[server.otel]
enabled = false
endpoint = "http://localhost:4317"  # OTLP gRPC endpoint
service_name = "athenas-studio"
sample_ratio = 1.0                  # 0.0 = no traces, 1.0 = all traces
```

## IP Filtering

Control access to the API server by IP address or CIDR range:

```toml
[server]
# Empty allowlist = allow all IPs
ip_allowlist = ["10.0.0.0/8", "192.168.1.100"]
ip_denylist = ["10.0.0.5"]
```

- **Allowlist** — If non-empty, only listed IPs/CIDRs can access the server
- **Denylist** — Listed IPs/CIDRs are always blocked, even if in allowlist
- Supports both individual IPs (`192.168.1.100`) and CIDR ranges (`10.0.0.0/8`)

## Environment Variables

The config file is the primary configuration method. However, logging can be controlled via:

- `RUST_LOG` — Override log level (e.g., `RUST_LOG=debug`)
- `ATHENAS_CONFIG` — Override config file path (defaults to `~/.athenas/config.toml`)
