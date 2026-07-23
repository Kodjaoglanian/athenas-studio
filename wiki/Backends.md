# Backends

Athenas Studio supports multiple inference backends, each optimized for different use cases.

## Backend Types

| Backend | Type | Best For | GPU Support |
|---------|------|----------|-------------|
| **llama.cpp** | Local subprocess | Single-user, GGUF, CPU/GPU mix, multimodal | CUDA, ROCm, Vulkan, Metal |
| **vLLM** | Python server | High-throughput, multi-user, PagedAttention | CUDA, ROCm |
| **auto** | Auto-detect | Automatic selection based on hardware | Depends |

## llama.cpp Backend

The llama.cpp backend runs `llama-server` as a subprocess and communicates via HTTP.

### Features

- **GGUF model format** — Quantized models (Q4_K_M, Q5_K_M, Q8_0, etc.)
- **GPU offloading** — Configurable GPU layers (`-1` for all)
- **Flash attention** — Faster inference on supported hardware
- **Multimodal** — Automatic mmproj detection for vision models
- **Reasoning mode** — Configurable thinking budget for Qwen3.5, DeepSeek R1
- **KV cache slots** — Parallel slots with save/restore/erase
- **Jinja templates** — Proper chat template rendering with `--jinja` flag

### Server Startup Flags

Athenas Studio automatically configures the llama-server with optimal flags:

```
llama-server \
  --model <path> \
  --ctx-size 4096 \
  --batch-size 512 \
  --threads <auto> \
  --port <random> \
  --host 127.0.0.1 \
  --parallel 4 \
  --cont-batching \
  --cache-prompt \
  --warmup \
  --jinja \
  --metrics \
  [--n-gpu-layers -1] \
  [--flash-attn] \
  [--mmap] \
  [--mlock] \
  [--mmproj <path>] \
  [--reasoning budget] \
  [--no-reasoning]  # fallback if --reasoning unsupported
```

### Auto-Download

If `llama-server` is not found in PATH, Athenas Studio automatically downloads the appropriate pre-built binary for your platform and architecture.

### Auto-Install Dependencies

If the llama-server binary fails to start due to missing shared libraries (e.g., `libgomp.so.1`), Athenas Studio attempts to install them automatically using the system's package manager.

### Hardware Detection

Athenas Studio detects:
- **CUDA** — NVIDIA GPUs
- **ROCm** — AMD GPUs
- **Vulkan** — Cross-platform GPU
- **Metal** — Apple Silicon
- **CPU** — Fallback

Based on detected hardware, it automatically:
- Sets optimal thread count (leaving 1 core free)
- Caps context size based on available RAM
- Adjusts batch size for memory safety
- Reserves RAM for OS (`ram_reserve_mb`)

## vLLM Backend

The vLLM backend runs a vLLM server for high-throughput inference.

### Features

- **PagedAttention** — Efficient memory management for large batches
- **Continuous batching** — Dynamic request batching
- **Tensor parallelism** — Multi-GPU support
- **High throughput** — Optimized for multi-user serving

### Setup

```bash
pip install vllm
```

### Usage

```bash
athenas serve model.gguf --backend vllm --port 8080
```

### Limitations

- Requires CUDA or ROCm (no CPU support)
- Larger memory footprint than llama.cpp
- No multimodal support (yet)
- No KV cache slot management

## Auto Backend

When `backend = "auto"` (default), Athenas Studio selects the best backend based on:

1. If vLLM is installed and CUDA/ROCm is available → vLLM
2. Otherwise → llama.cpp

## Benchmarking

Compare backend performance:

```bash
athenas backend benchmark
athenas backend benchmark --model model.gguf
```

This runs a standardized inference benchmark and reports:
- Tokens per second
- Time to first token
- Total generation time
- Memory usage

## Backend Factory

The `BackendFactory` in `athenas-inference` creates backend instances:

```rust
let backend = BackendFactory::create(BackendType::LlamaCpp, &hardware)?;
backend.load_model(load_config).await?;
```

The factory handles:
- Hardware detection
- Backend selection (auto mode)
- Binary discovery and auto-download
- Initial configuration
