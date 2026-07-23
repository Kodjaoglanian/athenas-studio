# Athenas Studio Wiki

> **A powerful CLI/TUI tool for running LLM models locally with CUDA, ROCm, and vLLM support. Compatible with HuggingFace model hub and OpenAI API.**

Athenas Studio is an enterprise-grade local LLM inference platform that provides a TUI (Terminal User Interface), CLI, and an OpenAI-compatible API server. It supports multiple backends (llama.cpp, vLLM), multi-model management, session management with KV cache persistence, and production features like rate limiting, metrics, and audit logging.

## Key Differentiators vs LM Studio

| Feature | Athenas Studio | LM Studio |
|---------|---------------|-----------|
| **TUI / Headless** | ✅ Full TUI + CLI + API | ❌ GUI only |
| **Session Management** | ✅ Server-side conversation history | ❌ None |
| **KV Cache Slots** | ✅ Save/restore/erase slot checkpoints | ❌ None |
| **Prometheus Metrics** | ✅ Native `/metrics` endpoint | ❌ None |
| **Embeddings API** | ✅ `/v1/embeddings` | ✅ Limited |
| **Function Calling** | ✅ Tool use with structured JSON | ✅ Limited |
| **Multi-tenant API Keys** | ✅ Per-key quotas, rate limits, model access | ❌ Single key |
| **Model Routing/Fallback** | ✅ Automatic failover chains | ❌ None |
| **Audit Logging** | ✅ JSONL audit trail + TUI logs page | ❌ None |
| **Reasoning/Thinking Mode** | ✅ Configurable budget (Qwen3.5, DeepSeek R1) | ✅ Limited |
| **Multimodal** | ✅ Auto mmproj detection | ✅ Yes |
| **Self-Update** | ✅ `athenas update` | ❌ Manual |

## Wiki Pages

- **[Installation](Installation)** — How to install Athenas Studio
- **[Quick Start](Quick-Start)** — Get up and running in 5 minutes
- **[Configuration](Configuration)** — Full config reference (`~/.athenas/config.toml`)
- **[CLI Commands](CLI-Commands)** — Complete CLI command reference
- **[TUI Guide](TUI-Guide)** — Terminal UI walkthrough
- **[API Reference](API-Reference)** — Complete REST API documentation
- **[Embeddings API](Embeddings-API)** — `/v1/embeddings` endpoint
- **[Function Calling](Function-Calling)** — Tool use and function calling
- **[Multi-tenant API Keys](Multi-tenant-API-Keys)** — API key management
- **[Model Routing](Model-Routing)** — Fallback chains and routing
- **[Audit Logging](Audit-Logging)** — Compliance audit trail
- **[Session & Slot Management](Session-and-Slot-Management)** — KV cache persistence
- **[Backends](Backends)** — llama.cpp and vLLM configuration
- **[Deployment](Deployment)** — Production deployment guide
- **[Architecture](Architecture)** — Internal codebase architecture
