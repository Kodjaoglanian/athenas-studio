# AGENTS.md — Project Guide for AI Agents

## Build & Test Commands

```bash
# Format check
cargo fmt

# Lint
cargo clippy

# Build (debug)
cargo build

# Build (release)
cargo build --release

# Run TUI
cargo run --release

# Run tests (currently minimal)
cargo test
```

## Architecture

Cargo workspace with 6 crates:

| Crate | Path | Responsibility |
|-------|------|----------------|
| `athenas-core` | `crates/athenas-core` | Config (`config.toml`), hardware detection, model registry, error types |
| `athenas-inference` | `crates/athenas-inference` | Backend trait, llama.cpp backend, vLLM backend, RemoteBackend, llama-server auto-download |
| `athenas-hub` | `crates/athenas-hub` | HuggingFace API client, download manager, mmproj auto-download |
| `athenas-server` | `crates/athenas-server` | OpenAI-compatible API server (axum), multi-model manager, file upload, vector store, tracing |
| `athenas-tui` | `crates/athenas-tui` | Terminal UI (ratatui + crossterm), 6 tabs (Chat, Models, Browser, Server, Settings, Logs) |
| `athenas-cli` | `crates/athenas-cli` | CLI entry point (clap), command dispatch |

## Key Files

| File | Purpose |
|------|---------|
| `crates/athenas-core/src/config.rs` | AppConfig struct, load/save to `~/.athenas/config.toml` |
| `crates/athenas-core/src/hardware.rs` | HardwareDetector — detects CPU, GPU (CUDA/ROCm/Vulkan/Metal), RAM |
| `crates/athenas-core/src/model_registry.rs` | ModelRegistry — scans `~/.athenas/models/` for .gguf/.safetensors files (filters out mmproj) |
| `crates/athenas-inference/src/llama_cpp.rs` | LlamaCppBackend — starts llama-server subprocess, GPU runtime selection, health polling |
| `crates/athenas-inference/src/backend_setup.rs` | Auto-download of llama-server binary from GitHub releases (platform + GPU aware) |
| `crates/athenas-tui/src/app.rs` | TuiApp — main event loop, key handling, server start/stop, model loading |
| `crates/athenas-tui/src/server_manager.rs` | Detached server process management (start/stop/check_running) |
| `crates/athenas-tui/src/settings.rs` | SettingsState — F5 settings page state |
| `crates/athenas-tui/src/server_panel.rs` | ServerPanelState — F4 server panel state |
| `crates/athenas-tui/src/chat.rs` | ChatState, ChatMessage — chat data structures (messages, streaming, system prompt) |
| `crates/athenas-tui/src/markdown.rs` | Markdown renderer — converts markdown to styled ratatui Lines (headers, bold, code blocks, lists, etc.) |
| `crates/athenas-tui/src/components.rs` | Rendering — chat area, status bar, model list, server panel, settings, logs |
| `crates/athenas-server/src/semantic_cache.rs` | SemanticCache — cosine similarity cache with TTL, LRU eviction, disk persistence |
| `crates/athenas-cli/src/commands/serve.rs` | `athenas serve` command — loads model and starts API server |

## Important Patterns

### Config Sync (Settings → App)

The Settings page (F5) has its own copy of `AppConfig` (`settings_state.config`). After any change, it must be synced back:
```rust
self.config = self.settings_state.config.clone();
```
Without this, `load_model()` uses stale values. This was a bug in v0.7.24.

### Detached Server

The server runs as a **separate process** (`athenas serve`), not in-process. The TUI:
1. Spawns `athenas serve` via `server_manager::start_detached()`
2. Polls `/v1/health` until the server responds (up to 5 minutes)
3. Saves state to `~/.athenas/server_state.json` for re-attachment

### GPU Binary Auto-Download

`backend_setup.rs` detects the OS and GPU, then downloads the correct llama-server binary:
- Linux: Vulkan binary (no CUDA prebuilt available)
- Windows: CUDA binary for NVIDIA, HIP for AMD
- macOS: Metal (built into all macOS binaries)

A `.llama-server-variant` marker file tracks which variant was installed.

### Blocking I/O

All file I/O (`config.save()`, `stop_by_pid()`) must use `tokio::task::spawn_blocking()` to avoid freezing the TUI.

### mmproj Files

Files with `mmproj` in the name are multimodal projectors (CLIP vision models), not standalone LLM models. They are:
- Filtered out of the model list (`scan_dir_for_models`)
- Rejected with a clear error if loaded directly
- Auto-detected and loaded via `--mmproj` flag when loading the main model

### Server Panel Status & Errors

`ServerPanelState` has an explicit `status_is_error` flag — set messages via
`set_status()` (info) / `set_error()` (red) / `clear_status()`. Never write
`status_message` directly and never infer errors from string prefixes.

### RAM Estimate (shared heuristic)

`athenas_core::estimate_model_ram_mb(model_size_mb, context_size)` =
`file_size + (ctx / 1024) * 64` MB. Used by BOTH the chat loader and the
server panel's `estimate_selected_model_load()`. If you change the heuristic,
keep it shared — don't duplicate the formula.

### Editing Fields (no placeholder leakage)

`ServerPanelState::edit_value()` (not `field_value()`) pre-fills the edit
buffer. `field_value()` returns *display* strings ("(none)", "unlimited",
masked secrets) that would corrupt the config if edited literally. Any new
field with a display placeholder MUST add a matching `edit_value()` arm.

### Memory Refresh

Available RAM is stale at startup. The server panel re-reads it every 5s
while the tab is open via `poll_mem_refresh()` (spawn_blocking → `detect_memory_mb`).
Use `HardwareInfo::refresh_memory()` or `detect_memory_mb()` for fresh data;
pre-flight checks in `start_server()` read memory before the estimate verdict.

### Chat Generation & Cancellation

Chat generation runs in a `tokio::spawn` background task. The TUI polls
chunks via `poll_chat_stream()` every 100ms (non-blocking `try_recv()`).

Cancellation uses `tokio::sync::watch::channel(false)`. The background task
wraps the stream in `tokio::select!` with `cancel_rx.changed()`. When the
user presses Esc, `cancel_generation()` sends `true` via the watch sender.

**Critical:** After creating the watch channel, call `borrow_and_update()`
to mark the initial value as seen. Without this, `changed()` fires
immediately and cancels the stream before it starts (was a bug in v0.8.5).

### Markdown Rendering

Assistant messages are rendered with markdown (`markdown.rs`). User and
system messages stay as plain text. During streaming, text renders as
plain text (avoids flickering from incomplete markdown) and markdown is
applied when the message finalizes via `finalize_streaming()`.

### Semantic Cache

The server-side semantic cache (`semantic_cache.rs`) caches chat completion
responses based on embedding similarity. It uses cosine similarity, TTL,
and LRU eviction with disk persistence to `~/.athenas/cache/`. Disabled
by default; enable via `[server.semantic_cache]` in config.toml.

## Release Process

1. Bump version in `Cargo.toml` and `README.md`
2. Commit with message `vX.Y.Z: <description>`
3. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. GitHub Actions builds and publishes binaries for all platforms
5. Create GitHub release with `gh release create`

## Common Pitfalls

- **Don't use `std::fs` in async context** — use `spawn_blocking`
- **Don't forget to sync settings** — `self.config = self.settings_state.config.clone()`
- **Don't load mmproj files directly** — they crash llama-server with "unsupported architecture: 'clip'"
- **Check `server_health_task` before starting** — not just `server_start_task`
- **Always kill stale processes** before starting a new server
- **The latest llama.cpp release may be incomplete** — search recent releases for the required asset
- **Don't pre-fill edit buffers from `field_value()`** — use `edit_value()` or placeholders leak into config
- **Don't write `server_panel_state.status_message` directly** — use `set_status()`/`set_error()` so the status bar colors correctly
