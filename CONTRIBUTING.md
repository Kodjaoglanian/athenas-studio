# Contributing to Athenas Studio

Thank you for your interest in contributing to Athenas Studio! This document outlines the process for contributing.

## Development Setup

1. **Install Rust** via [rustup](https://rustup.rs)
2. Clone the repository:
   ```bash
   git clone https://github.com/Kodjaoglanian/athenas-studio.git
   cd athenas-studio
   ```
3. Build:
   ```bash
   cargo build
   ```

## Architecture

The project is organized as a Cargo workspace with 6 crates:

| Crate | Description |
|-------|-------------|
| `athenas-core` | Config, storage, hardware detection, model registry, resource limits |
| `athenas-inference` | Backend trait, llama.cpp & vLLM implementations, mmproj auto-detection |
| `athenas-hub` | HuggingFace API client, download manager, mmproj auto-download |
| `athenas-server` | OpenAI-compatible API server (axum), multi-model manager, file upload |
| `athenas-tui` | Terminal UI (ratatui + crossterm), server panel with multi-model management |
| `athenas-cli` | CLI entry point (clap) |

## Key Features

- **Multi-Model Management** — Load/unload multiple models simultaneously via TUI or API
- **Multimodal Support** — Automatic mmproj detection and download for vision models
- **Reasoning Mode** — Configurable thinking budget for reasoning models (Qwen3.5, DeepSeek R1)
- **Auto Resource Limits** — Hardware-aware capping of threads, context, and batch size

## Coding Guidelines

- Follow Rust idioms and `clippy` suggestions
- Use `cargo fmt` before committing
- Keep changes focused — one feature/fix per PR
- Add tests for new functionality
- Document public APIs with `///` doc comments

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes following [Conventional Commits](https://www.conventionalcommits.org/)
4. Ensure `cargo test` and `cargo clippy` pass
5. Open a Pull Request with a clear description

## Reporting Issues

Use [GitHub Issues](https://github.com/Kodjaoglanian/athenas-studio/issues) to report bugs or request features. Include:

- OS and version
- Rust version (`rustc --version`)
- Hardware (GPU, CUDA/ROCm version)
- Steps to reproduce
- Expected vs actual behavior
