# Installation

## One-Line Install (Linux & macOS)

```bash
curl -fsSL https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.sh | bash
```

## One-Line Install (Windows PowerShell)

```powershell
irm https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.ps1 | iex
```

The installer automatically:
1. Detects your platform and architecture
2. Downloads the latest release binary
3. Verifies the SHA256 checksum
4. Installs to `~/.athenas/bin`
5. Adds `~/.athenas/bin` to your PATH

## Supported Platforms

| OS | Architecture | Target Triple |
|----|-------------|---------------|
| Linux | x86_64 | `x86_64-unknown-linux-gnu` |
| Linux | x86_64 (musl) | `x86_64-unknown-linux-musl` |
| Linux | ARM64 | `aarch64-unknown-linux-gnu` |
| Linux | ARM64 (musl) | `aarch64-unknown-linux-musl` |
| macOS (Intel) | x86_64 | `x86_64-apple-darwin` |
| macOS (Apple Silicon) | ARM64 | `aarch64-apple-darwin` |
| Windows | x86_64 | `x86_64-pc-windows-msvc` |
| Windows | ARM64 | `aarch64-pc-windows-msvc` |

## From Source

```bash
git clone https://github.com/Kodjaoglanian/athenas-studio.git
cd athenas-studio
cargo build --release
# Binary at target/release/athenas
```

### Prerequisites

- **Rust** 1.70+ (install via [rustup](https://rustup.rs))
- **llama.cpp** — Install `llama-server` and ensure it's in PATH (for llama.cpp backend)
  - Athenas Studio auto-downloads `llama-server` if not found
- **vLLM** — `pip install vllm` (for vLLM backend, requires CUDA or ROCm)

## Verify Installation

```bash
athenas --version
athenas hardware
```

## Update

```bash
athenas update
```

This downloads and installs the latest release, replacing the current binary.
