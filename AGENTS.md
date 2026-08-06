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
