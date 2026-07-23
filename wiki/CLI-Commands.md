# CLI Commands

## Global Flags

All commands support these optional flags:

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Enable info-level logging |
| `--debug` | `-d` | Enable debug-level logging |

## Commands

### `athenas` (default: TUI)

Start the Terminal User Interface.

```bash
athenas
```

### `athenas chat`

Start an interactive chat session in the terminal.

```bash
athenas chat model.gguf
athenas chat --backend llama.cpp --gpu-layers -1 --context-size 4096
```

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `auto` | Backend: `auto`, `llama.cpp`, `vllm` |
| `--gpu-layers` | `-1` | GPU layers to offload (-1 = all) |
| `--context-size` | `4096` | Context window size |

### `athenas serve`

Start the OpenAI-compatible API server.

```bash
athenas serve model.gguf --port 8080
athenas serve model.gguf --host 0.0.0.0 --port 8080 --backend vllm
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Bind port |
| `--backend` | `auto` | Backend type |
| `--gpu-layers` | `-1` | GPU layers |
| `--context-size` | `4096` | Context window |
| `--max-concurrent` | `10` | Max simultaneous requests |
| `--rate-limit` | `20` | Requests per second per IP |
| `--timeout` | `120` | Request timeout (seconds) |
| `--max-body-size` | `10` | Max body size (MB) |

### `athenas run`

One-shot inference — generate a response and exit.

```bash
athenas run model.gguf "What is the meaning of life?"
athenas run model.gguf "Explain quantum computing" --temperature 0.3 --max-tokens 512
```

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `auto` | Backend type |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-tokens` | `2048` | Max tokens to generate |
| `--gpu-layers` | `-1` | GPU layers |

### `athenas models`

Manage local models.

```bash
# List local models
athenas models list

# Search HuggingFace
athenas models search "llama 3" --gguf
athenas models search "mistral" --pipeline text-generation

# Download a model
athenas models pull TheBloke/Llama-2-7B-Chat-GGUF
athenas models pull TheBloke/Llama-2-7B-Chat-GGUF --file llama-2-7b-chat.Q4_K_M.gguf

# Show model info
athenas models info llama-2-7b-chat

# Remove a model
athenas models remove llama-2-7b-chat
```

### `athenas config`

Manage configuration.

```bash
athenas config show
athenas config get inference.default_backend
athenas config set inference.default_backend llama.cpp
athenas config init
```

### `athenas hardware`

Show detected hardware (CPU, GPU, memory).

```bash
athenas hardware
```

### `athenas backend`

List and benchmark backends.

```bash
athenas backend list
athenas backend benchmark
athenas backend benchmark --model model.gguf
```

### `athenas login`

Set HuggingFace access token for gated models.

```bash
athenas login --token hf_xxxxx
```

### `athenas update`

Update athenas to the latest release.

```bash
athenas update
```
