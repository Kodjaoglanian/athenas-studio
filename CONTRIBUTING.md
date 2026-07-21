# Contributing to Athenas Studio

Thank you for your interest in contributing to Athenas Studio! This document outlines the process for contributing.

## Development Setup

1. **Install Rust** via [rustup](https://rustup.rs)
2. Clone the repository:
   ```bash
   git clone https://github.com/Kodjaoglanian/athenas-studio.git
   cd athenas
   ```
3. Build:
   ```bash
   cargo build
   ```

## Architecture

The project is organized as a Cargo workspace with 6 crates:

| Crate | Description |
|-------|-------------|
| `athenas-core` | Config, storage, hardware detection, model registry |
| `athenas-inference` | Backend trait, llama.cpp & vLLM implementations |
| `athenas-hub` | HuggingFace API client, download manager |
| `athenas-server` | OpenAI-compatible API server (axum) |
| `athenas-tui` | Terminal UI (ratatui + crossterm) |
| `athenas-cli` | CLI entry point (clap) |

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
