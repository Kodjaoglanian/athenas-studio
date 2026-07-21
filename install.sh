#!/usr/bin/env bash
# Athenas Studio installer — one-line install: curl -fsSL https://athenas.studio/install.sh | bash
set -euo pipefail

REPO="Kodjaoglanian/athenas-studio"
INSTALL_DIR="${ATHENAS_INSTALL_DIR:-$HOME/.athenas/bin}"
CONFIG_DIR="${ATHENAS_CONFIG_DIR:-$HOME/.athenas}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }
success() { echo -e "${GREEN}[ok]${NC} $*"; }

# Detect platform
detect_target() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Linux)
            case "$arch" in
                x86_64|amd64)  echo "x86_64-unknown-linux-musl" ;;
                aarch64|arm64) echo "aarch64-unknown-linux-musl" ;;
                *) error "Unsupported architecture: $arch on Linux" ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64|amd64)  echo "x86_64-apple-darwin" ;;
                arm64|aarch64) echo "aarch64-apple-darwin" ;;
                *) error "Unsupported architecture: $arch on macOS" ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*)
            case "$arch" in
                x86_64|amd64)  echo "x86_64-pc-windows-msvc" ;;
                aarch64|arm64) echo "aarch64-pc-windows-msvc" ;;
                *) error "Unsupported architecture: $arch on Windows" ;;
            esac
            ;;
        *) error "Unsupported OS: $os" ;;
    esac
}

# Fetch latest release version
get_latest_version() {
    local api_url="https://api.github.com/repos/${REPO}/releases/latest"
    local resp

    if command -v curl &>/dev/null; then
        resp="$(curl -fsSL "$api_url")"
    elif command -v wget &>/dev/null; then
        resp="$(wget -qO- "$api_url")"
    else
        error "Neither curl nor wget found. Please install one."
    fi

    # Prefer jq for robust JSON parsing
    if command -v jq &>/dev/null; then
        echo "$resp" | jq -r '.tag_name'
    else
        # Fallback: extract tag_name with sed (handles both pretty and minified JSON)
        echo "$resp" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1
    fi
}

main() {
    echo ""
    echo -e "${CYAN}\033[1m    ___   __   ____  _   _ _____ _____ ____     ${NC}"
    echo -e "${CYAN}\033[1m   / _ \\\\ / /_ / ___|| | |  ___|_   _|  _ \\\\    ${NC}"
    echo -e "${CYAN}\033[1m  / /_\\\\_/ __|\\\\___ \\\\| |_| | |_    | | | |_) |   ${NC}"
    echo -e "${CYAN}\033[1m / /_\\\\  \\\\_| |____) |  _  |  _|   | | |  _ <    ${NC}"
    echo -e "${CYAN}\033[1m \\\\____|\\\\__|____/ |_| |_|_|     |_| |_| \\\\_\\\\   ${NC}"
    echo -e "${CYAN}\033[1m        Studio — Local LLM Inference${NC}"
    echo ""

    local target version archive_name download_url
    tmp_dir=""
    trap 'rm -rf "${tmp_dir:-}"' EXIT

    target="$(detect_target)"
    info "Detected target: ${target}"

    version="$(get_latest_version)"
    if [ -z "$version" ]; then
        error "Failed to fetch latest version. Check your internet connection."
    fi
    info "Latest version: ${version}"

    # Determine archive extension
    case "$target" in
        *windows*) archive_name="athenas-${version}-${target}.zip" ;;
        *)         archive_name="athenas-${version}-${target}.tar.gz" ;;
    esac

    download_url="https://github.com/${REPO}/releases/download/${version}/${archive_name}"
    info "Downloading: ${download_url}"

    tmp_dir="$(mktemp -d)"

    if command -v curl &>/dev/null; then
        curl -fsSL "$download_url" -o "${tmp_dir}/${archive_name}" || error "Download failed"
    else
        wget -qO "${tmp_dir}/${archive_name}" "$download_url" || error "Download failed"
    fi

    # Verify SHA256 if available
    local sha256_url="${download_url}.sha256"
    info "Verifying checksum..."
    if command -v curl &>/dev/null; then
        curl -fsSL "$sha256_url" -o "${tmp_dir}/${archive_name}.sha256" 2>/dev/null || true
    else
        wget -qO "${tmp_dir}/${archive_name}.sha256" "$sha256_url" 2>/dev/null || true
    fi

    if [ -f "${tmp_dir}/${archive_name}.sha256" ]; then
        (cd "$tmp_dir" && sha256sum -c "${archive_name}.sha256" 2>/dev/null) || warn "Checksum verification skipped (sha256sum not available or mismatch)"
        success "Checksum verified"
    else
        warn "No checksum available, skipping verification"
    fi

    # Extract
    info "Extracting..."
    case "$archive_name" in
        *.tar.gz)
            tar xzf "${tmp_dir}/${archive_name}" -C "$tmp_dir"
            ;;
        *.zip)
            if command -v unzip &>/dev/null; then
                unzip -o "${tmp_dir}/${archive_name}" -d "$tmp_dir"
            else
                error "unzip not found. Please install unzip."
            fi
            ;;
    esac

    # Install
    mkdir -p "$INSTALL_DIR"
    local binary_name="athenas"
    case "$target" in
        *windows*) binary_name="athenas.exe" ;;
    esac

    if [ -f "${tmp_dir}/${binary_name}" ]; then
        mv "${tmp_dir}/${binary_name}" "${INSTALL_DIR}/${binary_name}"
    elif [ -f "${tmp_dir}/athenas" ]; then
        mv "${tmp_dir}/athenas" "${INSTALL_DIR}/${binary_name}"
    elif [ -f "${tmp_dir}/athenas.exe" ]; then
        mv "${tmp_dir}/athenas.exe" "${INSTALL_DIR}/${binary_name}"
    else
        error "Binary not found in archive. Files: $(ls -la $tmp_dir)"
    fi

    chmod +x "${INSTALL_DIR}/${binary_name}" 2>/dev/null || true
    success "Installed to: ${INSTALL_DIR}/${binary_name}"

    # Create config directory
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${CONFIG_DIR}/models"
    mkdir -p "${CONFIG_DIR}/cache"
    mkdir -p "${CONFIG_DIR}/data"
    success "Config directory: ${CONFIG_DIR}"

    # Add to PATH
    local shell_rc=""
    case "$(basename "$SHELL")" in
        bash) shell_rc="$HOME/.bashrc" ;;
        zsh)  shell_rc="$HOME/.zshrc" ;;
        fish) shell_rc="$HOME/.config/fish/config.fish" ;;
        *)    shell_rc="$HOME/.profile" ;;
    esac

    local path_line="export PATH=\"${INSTALL_DIR}:\$PATH\""
    local fish_line="set -gx PATH ${INSTALL_DIR} \$PATH"

    if [ "$(basename "$SHELL")" = "fish" ]; then
        if ! grep -q "$INSTALL_DIR" "$shell_rc" 2>/dev/null; then
            echo "$fish_line" >> "$shell_rc"
            success "Added to PATH in ${shell_rc}"
        fi
    else
        if ! grep -q "$INSTALL_DIR" "$shell_rc" 2>/dev/null; then
            echo "$path_line" >> "$shell_rc"
            success "Added to PATH in ${shell_rc}"
        fi
    fi

    # Verify
    echo ""
    if "${INSTALL_DIR}/${binary_name}" --version 2>/dev/null; then
        echo ""
        success "Athenas Studio installed successfully!"
        echo ""
        echo "  Run 'athenas --help' to get started"
        echo "  Or start the TUI with 'athenas'"
        echo ""
        if [ -n "$shell_rc" ]; then
            warn "Restart your shell or run: source ${shell_rc}"
        fi
        echo ""
    else
        warn "Installation complete, but binary verification failed."
        warn "Try running: ${INSTALL_DIR}/${binary_name} --help"
    fi
}

main "$@"
