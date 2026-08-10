use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use athenas_core::{AthenasError, HardwareInfo, Result};

use crate::backend::{Backend, ModelInfo};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, EmbeddingData,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, InferenceStats, MessageContent,
    ModelLoadConfig, Role, StreamChunk,
};

/// llama.cpp backend — uses llama.cpp server subprocess for inference
pub struct LlamaCppBackend {
    hardware: HardwareInfo,
    loaded: bool,
    model_path: String,
    model_name: String,
    context_size: u32,
    gpu_layers: i32,
    server_handle: Option<tokio::process::Child>,
    server_port: u16,
    client: reqwest::Client,
    /// Set to true if --reasoning flag caused server to fail, so we skip it on retry
    skip_reasoning_flag: bool,
    /// Whether reasoning/thinking mode is enabled (from config)
    reasoning_enabled: bool,
    /// Watchdog task that periodically checks if llama-server is alive
    watchdog_handle: Option<tokio::task::JoinHandle<()>>,
    /// Shared flag to signal the watchdog to stop
    watchdog_stop: Arc<std::sync::atomic::AtomicBool>,
}

impl LlamaCppBackend {
    pub fn new(hardware: &HardwareInfo) -> Self {
        Self {
            hardware: hardware.clone(),
            loaded: false,
            model_path: String::new(),
            model_name: String::new(),
            context_size: 4096,
            gpu_layers: -1,
            server_handle: None,
            server_port: 0,
            // Timeout for requests to llama-server. Prompt processing can
            // be slow on some hardware (e.g. 18 tokens/sec for a 2000-token
            // prompt = 111s). 300s gives enough headroom for large prompts
            // while still catching genuinely stuck requests.
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .connect_timeout(std::time::Duration::from_secs(10))
                .http1_only()
                .build()
                .unwrap(),
            skip_reasoning_flag: false,
            reasoning_enabled: true,
            watchdog_handle: None,
            watchdog_stop: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    fn find_llama_server(&self) -> Option<String> {
        let home = std::env::var("HOME").unwrap_or_default();

        // 1. Check ~/.athenas/bin first (our auto-installed version)
        let athenas_path = format!("{}/.athenas/bin/llama-server", home);
        if std::path::Path::new(&athenas_path).exists() {
            return Some(athenas_path);
        }

        // 2. Check PATH
        for cmd in &["llama-server", "llama_server", "server"] {
            if which::which(cmd).is_ok() {
                return Some(cmd.to_string());
            }
        }

        // 3. Check common install locations
        let candidates = [
            format!("{}/.local/bin/llama-server", home),
            "/usr/local/bin/llama-server".to_string(),
            "/usr/bin/llama-server".to_string(),
            "/opt/llama.cpp/build/bin/llama-server".to_string(),
        ];

        for path in &candidates {
            if std::path::Path::new(path).exists() {
                return Some(path.clone());
            }
        }

        None
    }

    async fn start_server(&mut self, config: &ModelLoadConfig) -> Result<()> {
        let server_bin = if let Some(path) = self.find_llama_server() {
            // Validate: check for shared libs next to the binary on Linux/macOS
            let p = std::path::Path::new(&path);
            if let Some(parent) = p.parent() {
                let needs_lib = std::env::consts::OS == "linux" || std::env::consts::OS == "macos";
                let has_lib = if std::env::consts::OS == "linux" {
                    std::fs::read_dir(parent)
                        .map(|entries| {
                            entries
                                .filter_map(|e| e.ok())
                                .any(|e| e.file_name().to_string_lossy().starts_with("libllama"))
                        })
                        .unwrap_or(false)
                } else if std::env::consts::OS == "macos" {
                    std::fs::read_dir(parent)
                        .map(|entries| {
                            entries
                                .filter_map(|e| e.ok())
                                .any(|e| e.file_name().to_string_lossy().ends_with(".dylib"))
                        })
                        .unwrap_or(false)
                } else {
                    true
                };

                // Check if GPU variant is needed but binary is CPU-only
                let needs_gpu =
                    config.gpu_layers != 0 && config.gpu_runtime != athenas_core::GpuRuntime::Cpu;
                let variant_marker = parent.join(".llama-server-variant");
                let current_variant = std::fs::read_to_string(&variant_marker).unwrap_or_default();
                // No marker = old install (before v0.7.22) = CPU-only binary
                let variant_is_cpu = current_variant.trim().is_empty()
                    || current_variant.contains("bin-ubuntu-x64")
                    || current_variant.contains("bin-win-cpu");
                let needs_redownload_for_gpu =
                    needs_gpu && variant_is_cpu && path.contains(".athenas");

                if needs_redownload_for_gpu {
                    info!(
                        "llama-server is CPU-only but GPU is configured (gpu_layers={}, runtime={:?}) — \
                         re-downloading with GPU support...",
                        config.gpu_layers, config.gpu_runtime
                    );
                    let new_path = crate::backend_setup::force_redownload_llama_server().await?;
                    new_path.to_string_lossy().to_string()
                } else if needs_lib && !has_lib {
                    // Binary exists but shared libs are missing — force re-download
                    info!("llama-server found but shared libs missing, re-downloading...");
                    // Only delete if it's in our bin dir (don't touch system installs)
                    if path.contains(".athenas") {
                        let _ = std::fs::remove_file(&path);
                        // Also clean up old .so files
                        if let Ok(entries) = std::fs::read_dir(parent) {
                            for entry in entries.flatten() {
                                let name = entry.file_name().to_string_lossy().to_string();
                                if name.starts_with("libllama") || name.starts_with("libggml") {
                                    let _ = std::fs::remove_file(entry.path());
                                }
                            }
                        }
                    }
                    let new_path = crate::backend_setup::ensure_llama_server().await?;
                    new_path.to_string_lossy().to_string()
                } else {
                    path
                }
            } else {
                path
            }
        } else {
            info!("llama-server not found, auto-downloading...");
            let path = crate::backend_setup::ensure_llama_server().await?;
            path.to_string_lossy().to_string()
        };

        self.server_port = find_free_port();

        // Store reasoning config for use in chat requests
        self.reasoning_enabled = config.reasoning_enabled;

        let mut cmd = tokio::process::Command::new(&server_bin);

        // Set LD_LIBRARY_PATH to the directory containing llama-server
        // so it can find shared libraries (libllama-server-impl.so, etc.)
        if let Some(parent) = std::path::Path::new(&server_bin).parent() {
            let lib_path = parent.to_string_lossy().to_string();
            let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let new_ld_path = if existing.is_empty() {
                lib_path
            } else {
                format!("{}:{}", lib_path, existing)
            };
            cmd.env("LD_LIBRARY_PATH", new_ld_path);
        }

        // === GPU runtime and device selection ===
        // Set environment variables based on the chosen GPU runtime and
        // device index. This controls which GPU the subprocess uses.
        //
        // IMPORTANT: On Linux, there is no CUDA prebuilt binary from
        // llama.cpp releases. We download the Vulkan binary instead,
        // which works with NVIDIA, AMD, and Intel GPUs. So when auto-
        // detecting on Linux, we prefer Vulkan over CUDA even if
        // nvidia-smi is available, because the binary we have is Vulkan.
        let effective_runtime = match config.gpu_runtime {
            athenas_core::GpuRuntime::Auto => {
                if cfg!(target_os = "macos") {
                    // macOS: Metal is built into all binaries
                    athenas_core::GpuRuntime::Metal
                } else if cfg!(target_os = "linux") {
                    // Linux: we download Vulkan binary (no CUDA prebuilt)
                    // Prefer Vulkan over ROCm because:
                    // 1. We download the Vulkan binary, not ROCm
                    // 2. Many AMD APUs (Barcelo, Renoir, etc.) have rocm-smi
                    //    but don't actually support ROCm compute
                    // 3. Vulkan works with NVIDIA, AMD, and Intel
                    if self.hardware.has_vulkan {
                        athenas_core::GpuRuntime::Vulkan
                    } else if self.hardware.has_cuda {
                        // CUDA detected but no Vulkan — custom CUDA build
                        athenas_core::GpuRuntime::Cuda
                    } else if self.hardware.has_rocm {
                        // ROCm without Vulkan — unlikely but handle it
                        athenas_core::GpuRuntime::Rocm
                    } else {
                        athenas_core::GpuRuntime::Cpu
                    }
                } else {
                    // Windows and others: prefer CUDA > ROCm > Vulkan > CPU
                    if self.hardware.has_cuda {
                        athenas_core::GpuRuntime::Cuda
                    } else if self.hardware.has_rocm {
                        athenas_core::GpuRuntime::Rocm
                    } else if self.hardware.has_vulkan {
                        athenas_core::GpuRuntime::Vulkan
                    } else {
                        athenas_core::GpuRuntime::Cpu
                    }
                }
            }
            other => other,
        };

        if let Some(device) = config.gpu_device {
            match effective_runtime {
                athenas_core::GpuRuntime::Cuda => {
                    cmd.env("CUDA_VISIBLE_DEVICES", device.to_string());
                }
                athenas_core::GpuRuntime::Rocm => {
                    // ROCm uses HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES
                    cmd.env("HIP_VISIBLE_DEVICES", device.to_string());
                    cmd.env("ROCR_VISIBLE_DEVICES", device.to_string());
                }
                athenas_core::GpuRuntime::Vulkan => {
                    // Vulkan doesn't have a standard env var for device
                    // selection, but we can try GPU_SELECT_USE_FIRST_DEVICE
                    // as a hint. The llama.cpp Vulkan backend typically
                    // uses device 0 by default.
                    cmd.env("GGML_VULKAN_DEVICE", device.to_string());
                }
                _ => {}
            }
        }

        // When runtime is explicitly CPU, force gpu_layers to 0
        let effective_gpu_layers = if effective_runtime == athenas_core::GpuRuntime::Cpu {
            0
        } else {
            config.gpu_layers
        };

        tracing::info!(
            "GPU config: runtime={:?} (effective={:?}), device={:?}, gpu_layers={} (effective={})",
            config.gpu_runtime,
            effective_runtime,
            config.gpu_device,
            config.gpu_layers,
            effective_gpu_layers
        );

        cmd.arg("--model")
            .arg(&config.model_path)
            .arg("--ctx-size")
            .arg(config.context_size.to_string())
            .arg("--batch-size")
            .arg(config.batch_size.to_string())
            .arg("--threads")
            .arg(config.threads.to_string())
            .arg("--port")
            .arg(self.server_port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--parallel")
            .arg(config.parallel_slots.to_string())
            // Server read/write timeout — closes idle connections after 60s.
            // This does NOT cancel in-flight inference (llama_decode can't be
            // interrupted), but it frees up slots from clients that disconnected.
            .arg("--timeout")
            .arg("60")
            // Enterprise performance flags
            .arg("--cont-batching")
            .arg("--cache-prompt")
            .arg("--warmup")
            .arg("--jinja")
            .arg("--metrics");

        // Reasoning/thinking mode — configurable per model.
        // Models like Qwen3.5 can hang or produce extremely long thinking
        // traces. Use --reasoning off and --reasoning-budget 0 when disabled.
        // Skip if these flags caused a previous failure (unsupported version).
        if !self.skip_reasoning_flag {
            if !config.reasoning_enabled {
                cmd.arg("--reasoning").arg("off");
                cmd.arg("--reasoning-budget").arg("0");
            } else if config.reasoning_budget >= 0 {
                cmd.arg("--reasoning-budget")
                    .arg(config.reasoning_budget.to_string());
            }
        }

        if effective_gpu_layers >= 0 {
            cmd.arg("--n-gpu-layers")
                .arg(effective_gpu_layers.to_string());
        } else if self.hardware.has_cuda
            || self.hardware.has_rocm
            || self.hardware.has_vulkan
            || self.hardware.has_metal
        {
            // Auto-offload all layers to GPU when gpu_layers is -1 and a
            // supported GPU backend (CUDA, ROCm, Vulkan, or Metal) is
            // available. The llama.cpp binary must have been compiled with
            // the corresponding backend enabled.
            cmd.arg("--n-gpu-layers").arg("999");
        }

        if config.flash_attention {
            cmd.arg("--flash-attn").arg("on");
        }

        if config.use_mmap {
            cmd.arg("--mmap");
        }

        if config.use_mlock {
            cmd.arg("--mlock");
        }

        // Multimodal projector (mmproj) for vision models
        let mmproj_path = config
            .mmproj_path
            .clone()
            .or_else(|| auto_detect_mmproj(&config.model_path));
        if let Some(mmproj) = mmproj_path {
            info!("Using mmproj: {}", mmproj);
            cmd.arg("--mmproj").arg(mmproj);
        }

        // LoRA adapters
        for lora_path in &config.lora_paths {
            if !lora_path.is_empty() {
                info!("Loading LoRA adapter: {}", lora_path);
                cmd.arg("--lora").arg(lora_path);
            }
        }

        // Redirect llama-server stdout/stderr to a log file instead of
        // piping. When using Stdio::piped(), the pipe buffer (64KB) can
        // fill up if we don't drain it, causing the llama-server to
        // BLOCK on write() and become unresponsive. By writing to a
        // file, the llama-server can always write without blocking.
        let llama_log_path = {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            let dir = format!("{}/.athenas", home);
            let _ = std::fs::create_dir_all(&dir);
            format!("{}/llama-server.log", dir)
        };
        let llama_log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&llama_log_path)
            .map_err(|e| {
                AthenasError::Backend(format!("Failed to open llama-server log: {}", e))
            })?;
        let llama_log_clone = llama_log
            .try_clone()
            .map_err(|e| AthenasError::Backend(format!("Failed to dup log fd: {}", e)))?;

        info!("llama-server logs: {}", llama_log_path);

        cmd.stdout(std::process::Stdio::from(llama_log))
            .stderr(std::process::Stdio::from(llama_log_clone))
            .kill_on_drop(true);

        info!(
            "Starting llama-server on port {} with model: {} (gpu_layers={}, runtime={})",
            self.server_port, config.model_path, effective_gpu_layers, effective_runtime
        );

        let child = cmd
            .spawn()
            .map_err(|e| AthenasError::Backend(format!("Failed to start llama-server: {}", e)))?;

        self.server_handle = Some(child);

        // Wait for server to be ready
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // Poll for up to 120 seconds (240 attempts * 500ms interval).
        // Large models with GPU offload can take a while to load,
        // especially on Vulkan where shader compilation is slow.
        for _attempt in 0..240 {
            // Check if process exited early
            if let Some(ref mut child) = self.server_handle {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        // Read llama-server log file for diagnostic info
                        // (stderr is now redirected to a file, not piped)
                        let stderr_msg = std::fs::read_to_string(&llama_log_path)
                            .unwrap_or_default()
                            .lines()
                            .last()
                            .unwrap_or("")
                            .to_string();

                        let mut msg = format!("llama-server exited early with status: {}", status);
                        if !stderr_msg.is_empty() {
                            msg.push_str(&format!("\nlast log line: {}", stderr_msg));
                        }

                        // Check if --reasoning flags are unsupported by this version
                        let full_log = std::fs::read_to_string(&llama_log_path).unwrap_or_default();
                        if (full_log.contains("reasoning") || full_log.contains("unrecognized"))
                            && !self.skip_reasoning_flag
                        {
                            info!("--reasoning flag not supported, retrying without it...");
                            self.skip_reasoning_flag = true;
                            if let Some(ref mut child) = self.server_handle {
                                let _ = child.kill().await;
                            }
                            self.server_handle = None;
                            return self.retry_start_server(config).await;
                        }

                        if status.code() == Some(127) {
                            // Check if it's libgomp missing — try to auto-install
                            if full_log.contains("libgomp.so.1") {
                                info!("libgomp.so.1 missing, attempting auto-install...");
                                let installed = try_install_libgomp().await;
                                if installed {
                                    info!(
                                        "libgomp1 installed successfully, retrying server start..."
                                    );
                                    // Kill the failed child and retry
                                    if let Some(ref mut child) = self.server_handle {
                                        let _ = child.kill().await;
                                    }
                                    self.server_handle = None;
                                    // Retry the spawn by returning a special error
                                    // that the caller can retry, or just retry inline
                                    return self.retry_start_server(config).await;
                                }
                            }
                            // Check if libvulkan is missing (Vulkan binary needs it)
                            if full_log.contains("libvulkan.so") || full_log.contains("libvulkan") {
                                msg.push_str(
                                    "\n\nHint: Vulkan library missing. The GPU-accelerated \
                                     binary needs Vulkan libraries.\n\
                                     On Ubuntu/Debian: apt install -y libvulkan1 mesa-vulkan-drivers\n\
                                     On Fedora: dnf install -y vulkan-loader\n\
                                     On Arch: pacman -S vulkan-icd-loader\n\
                                     \n\
                                     For NVIDIA: also install nvidia-vulkan-icd or ensure \
                                     your NVIDIA driver supports Vulkan (driver >= 390.x).",
                                );
                            }
                            msg.push_str(
                                "\n\nHint: exit code 127 usually means the binary has missing \
                                 shared libraries. Try running 'ldd <path>' to check.\n\
                                 On Ubuntu/Debian: apt install -y libgomp1",
                            );
                        }
                        return Err(AthenasError::Backend(msg));
                    }
                    Ok(None) => {}
                    Err(_) => {}
                }
            }

            // Try health check with timeout
            let url = format!("http://127.0.0.1:{}/health", self.server_port);
            let health_req = self
                .client
                .get(&url)
                .timeout(tokio::time::Duration::from_secs(2));
            if let Ok(resp) = health_req.send().await {
                if resp.status().is_success() {
                    info!("llama-server is ready on port {}", self.server_port);
                    // Start watchdog to monitor llama-server health
                    self.start_watchdog();
                    return Ok(());
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        // Kill the server if it didn't start in time
        if let Some(ref mut child) = self.server_handle {
            let _ = child.kill().await;
        }
        self.server_handle = None;

        Err(AthenasError::Backend(
            "llama-server failed to start within 120 seconds".to_string(),
        ))
    }

    async fn retry_start_server(&mut self, config: &ModelLoadConfig) -> Result<()> {
        // Wait a moment for the package manager to settle
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        // Re-run start_server with the same config (boxed to allow async recursion)
        Box::pin(self.start_server(config)).await
    }

    /// Start a background watchdog task that periodically checks if
    /// the llama-server is still alive and responding. If the server
    /// becomes unresponsive (deadlock, OOM, crash), the watchdog logs
    /// an error and marks the backend as unloaded.
    fn start_watchdog(&mut self) {
        // Stop any existing watchdog
        self.watchdog_stop
            .store(true, std::sync::atomic::Ordering::SeqCst);

        self.watchdog_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag = self.watchdog_stop.clone();
        let port = self.server_port;
        let client = self.client.clone();

        self.watchdog_handle = Some(tokio::spawn(async move {
            let mut consecutive_failures = 0u32;
            loop {
                // Check every 30 seconds
                tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;

                if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }

                // Health check with short timeout
                let url = format!("http://127.0.0.1:{}/health", port);
                let result = client
                    .get(&url)
                    .timeout(tokio::time::Duration::from_secs(5))
                    .send()
                    .await;

                match result {
                    Ok(resp) if resp.status().is_success() => {
                        if consecutive_failures > 0 {
                            info!(
                                "llama-server recovered after {} failures",
                                consecutive_failures
                            );
                        }
                        consecutive_failures = 0;
                    }
                    Ok(resp) => {
                        consecutive_failures += 1;
                        warn!(
                            "llama-server health check returned {} (failure #{})",
                            resp.status(),
                            consecutive_failures
                        );
                    }
                    Err(e) => {
                        consecutive_failures += 1;
                        if consecutive_failures >= 3 {
                            error!(
                                "llama-server unresponsive after 3 consecutive failures ({}). \
                                 It may be deadlocked or out of memory. \
                                 Consider restarting the server.",
                                e
                            );
                        } else {
                            warn!(
                                "llama-server health check failed ({}): {} (failure #{})",
                                consecutive_failures, e, consecutive_failures
                            );
                        }
                    }
                }
            }
        }));
    }

    async fn stop_server(&mut self) -> Result<()> {
        // Stop the watchdog
        self.watchdog_stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }

        if let Some(mut child) = self.server_handle.take() {
            child.kill().await.map_err(|e| {
                AthenasError::Backend(format!("Failed to kill llama-server: {}", e))
            })?;
            info!("llama-server stopped");
        }
        Ok(())
    }

    fn server_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.server_port)
    }

    /// Quick check if the llama-server is alive and can process inference.
    /// Sends a minimal completion request (1 token) with a short timeout.
    /// This is more reliable than checking /health, which responds even
    /// when all inference slots are stuck (llama.cpp issue #7071).
    async fn check_server_alive(&self) -> bool {
        if self.server_port == 0 {
            return false;
        }
        // Try a minimal tokenization request first (fast, doesn't need a slot)
        let url = format!("http://127.0.0.1:{}/tokenize", self.server_port);
        let body = serde_json::json!({"content": "test"});
        match self
            .client
            .post(&url)
            .json(&body)
            .timeout(tokio::time::Duration::from_secs(3))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => true,
            _ => {
                // Fallback: try /health endpoint
                let health_url = format!("http://127.0.0.1:{}/health", self.server_port);
                self.client
                    .get(&health_url)
                    .timeout(tokio::time::Duration::from_secs(2))
                    .send()
                    .await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false)
            }
        }
    }
}

/// Try to install libgomp1 (GNU OpenMP) — needed by llama-server on some systems
async fn try_install_libgomp() -> bool {
    // Detect package manager and install
    let managers = [
        ("apt-get", vec!["apt-get", "install", "-y", "libgomp1"]),
        ("dnf", vec!["dnf", "install", "-y", "libgomp"]),
        ("yum", vec!["yum", "install", "-y", "libgomp"]),
        ("pacman", vec!["pacman", "-S", "--noconfirm", "gcc-libs"]),
        ("apk", vec!["apk", "add", "libgomp"]),
    ];

    for (name, args) in &managers {
        // Check if the package manager exists
        let check = tokio::process::Command::new("which")
            .arg(name)
            .output()
            .await;

        if let Ok(check_output) = check {
            if !check_output.status.success() {
                continue;
            }

            info!("Installing libgomp via {}...", name);
            // For apt-get, run update first
            if *name == "apt-get" {
                let _ = tokio::process::Command::new("apt-get")
                    .arg("update")
                    .arg("-qq")
                    .output()
                    .await;
            }

            let result = tokio::process::Command::new(args[0])
                .args(&args[1..])
                .output()
                .await;

            return match result {
                Ok(output) => {
                    if output.status.success() {
                        info!("libgomp installed successfully via {}", name);
                        true
                    } else {
                        tracing::warn!(
                            "Failed to install libgomp via {}: {}",
                            name,
                            String::from_utf8_lossy(&output.stderr)
                        );
                        false
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to run {}: {}", name, e);
                    false
                }
            };
        }
    }

    tracing::warn!("No supported package manager found to install libgomp");
    false
}

#[async_trait]
impl Backend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    async fn load_model(&mut self, config: ModelLoadConfig) -> Result<()> {
        self.model_path = config.model_path.clone();
        self.model_name = std::path::Path::new(&config.model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        self.context_size = config.context_size;
        self.gpu_layers = config.gpu_layers;

        self.start_server(&config).await?;
        self.loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.stop_server().await?;
        self.loaded = false;
        self.model_path.clear();
        self.model_name.clear();
        Ok(())
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        // Quick health check — if the llama-server is not responding,
        // fail fast instead of hanging for 120s.
        if !self.check_server_alive().await {
            error!("llama-server is not responding — inference impossible");
            return Err(AthenasError::Backend(
                "llama-server is not responding. Please restart the server.".to_string(),
            ));
        }

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let content = match &m.content {
                    MessageContent::Text(s) => serde_json::Value::String(s.clone()),
                    MessageContent::Parts(parts) => {
                        serde_json::to_value(parts).unwrap_or(serde_json::Value::Null)
                    }
                };
                serde_json::json!({"role": role, "content": content})
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "stream": false,
            "stop": request.stop.as_deref().unwrap_or(&[]),
            "chat_template_kwargs": {"enable_thinking": self.reasoning_enabled},
        });

        if let Some(ref tools) = request.tools {
            body["tools"] = tools.clone();
        }
        if let Some(ref tc) = request.tool_choice {
            body["tool_choice"] = tc.clone();
        }
        if let Some(ref rf) = request.response_format {
            body["response_format"] = rf.clone();
        }
        if let Some(ref grammar) = request.grammar {
            body["grammar"] = serde_json::Value::String(grammar.clone());
        }

        let url = format!("{}/v1/chat/completions", self.server_url());
        tracing::info!("Sending chat request to llama-server at {}", url);
        // Use the reqwest client's built-in timeout (120s).
        // Do NOT add a shorter tokio::time::timeout here — prompt processing
        // can legitimately take 60-90s for large prompts on slow hardware
        // (e.g. 18 tokens/sec × 2000 tokens = 111s). A 30s timeout would
        // cancel requests that would have succeeded.
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                error!("llama-server chat request failed: {}", e);
                AthenasError::Backend(format!("Request failed: {}", e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            error!("llama-server returned {} for chat: {}", status, text);
            return Err(AthenasError::Backend(format!(
                "llama-server returned {}: {}",
                status, text
            )));
        }

        let result: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AthenasError::Backend(format!("Invalid response: {}", e)))?;

        let content = result
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // If content is empty, try reasoning_content (Qwen3.5 thinking mode)
        let content = if content.is_empty() {
            result
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("reasoning_content"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            content
        };

        let usage = result.get("usage");
        let tokens_prompt = usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let tokens_generated = usage
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let tps = result
            .get("timings")
            .and_then(|t| t.get("tokens_per_second"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let stats = InferenceStats {
            tokens_generated,
            tokens_prompt,
            time_total_ms: 0,
            tokens_per_second: tps,
        };

        // Parse tool_calls from response (function calling)
        let tool_calls = result
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|tc| {
                        let id = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("call_0")
                            .to_string();
                        let call_type = tc
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("function")
                            .to_string();
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}")
                            .to_string();
                        crate::types::ToolCall {
                            id,
                            call_type,
                            function: crate::types::ToolCallFunction { name, arguments },
                        }
                    })
                    .collect::<Vec<_>>()
            });

        let finish_reason = result
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ChatResponse {
            model: self.model_name.clone(),
            message: ChatMessage::assistant(content),
            stats,
            tool_calls,
            finish_reason,
        })
    }

    async fn chat_stream(&self, request: ChatRequest, tx: mpsc::Sender<StreamChunk>) -> Result<()> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        // Quick health check — if the llama-server is not responding,
        // fail fast instead of hanging for 120s.
        if !self.check_server_alive().await {
            error!("llama-server is not responding — stream impossible");
            return Err(AthenasError::Backend(
                "llama-server is not responding. Please restart the server.".to_string(),
            ));
        }

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let content = match &m.content {
                    MessageContent::Text(s) => serde_json::Value::String(s.clone()),
                    MessageContent::Parts(parts) => {
                        serde_json::to_value(parts).unwrap_or(serde_json::Value::Null)
                    }
                };
                serde_json::json!({"role": role, "content": content})
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "stream": true,
            "stream_options": {"include_usage": true},
            "timings_per_token": true,
            "stop": request.stop.as_deref().unwrap_or(&[]),
            "chat_template_kwargs": {"enable_thinking": self.reasoning_enabled},
        });

        if let Some(ref tools) = request.tools {
            body["tools"] = tools.clone();
        }
        if let Some(ref tc) = request.tool_choice {
            body["tool_choice"] = tc.clone();
        }
        if let Some(ref rf) = request.response_format {
            body["response_format"] = rf.clone();
        }
        if let Some(ref grammar) = request.grammar {
            body["grammar"] = serde_json::Value::String(grammar.clone());
        }

        let url = format!("{}/v1/chat/completions", self.server_url());
        tracing::info!("Sending stream request to llama-server at {}", url);
        // Use the reqwest client's built-in timeout (120s).
        // Prompt processing can take 60-90s for large prompts — don't
        // add a shorter timeout that would cancel valid requests.
        let resp = self
            .client
            .post(&url)
            .header("Accept-Encoding", "identity")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                error!("llama-server stream request failed: {}", e);
                AthenasError::Backend(format!("Request failed: {}", e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            error!("llama-server returned {} for stream: {}", status, text);
            return Err(AthenasError::Backend(format!(
                "llama-server returned {}: {}",
                status, text
            )));
        }

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut full_text = String::new();
        let start_time = std::time::Instant::now();
        let mut token_count: u32 = 0;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    buffer.push_str(&String::from_utf8_lossy(&chunk));
                }
                Err(e) => {
                    tracing::warn!("Stream chunk error (graceful handling): {}", e);
                    // If we received ANY tokens (content or reasoning), finalize
                    // gracefully instead of returning an error. Small models
                    // may only produce reasoning before the connection drops.
                    if token_count > 0 {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        let tps = if elapsed > 0.0 {
                            token_count as f32 / elapsed
                        } else {
                            0.0
                        };
                        let _ = tx
                            .send(StreamChunk {
                                text: String::new(),
                                done: true,
                                is_reasoning: false,
                                stats: Some(InferenceStats {
                                    tokens_generated: token_count,
                                    tokens_prompt: 0,
                                    time_total_ms: (elapsed * 1000.0) as u64,
                                    tokens_per_second: tps,
                                }),
                            })
                            .await;
                        return Ok(());
                    }
                    return Err(AthenasError::Backend(format!("Stream error: {}", e)));
                }
            }

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    let tps = if elapsed > 0.0 {
                        token_count as f32 / elapsed
                    } else {
                        0.0
                    };
                    let _ = tx
                        .send(StreamChunk {
                            text: String::new(),
                            done: true,
                            is_reasoning: false,
                            stats: Some(InferenceStats {
                                tokens_generated: token_count,
                                tokens_prompt: 0,
                                time_total_ms: (elapsed * 1000.0) as u64,
                                tokens_per_second: tps,
                            }),
                        })
                        .await;
                    return Ok(());
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let choices = json.get("choices").and_then(|c| c.get(0));
                    let delta = choices.and_then(|c| c.get("delta"));

                    // Read both content and reasoning_content.
                    // Qwen3.5 and similar models put thinking tokens in
                    // reasoning_content — if we only read content, the model
                    // appears to hang while it generates internal reasoning.
                    let content = delta
                        .and_then(|d| d.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let reasoning = delta
                        .and_then(|d| d.get("reasoning_content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    // Send reasoning tokens (if any) with a visual prefix
                    // so the user sees the model is working.
                    if !reasoning.is_empty() {
                        token_count += 1;
                        let elapsed = start_time.elapsed().as_secs_f32();
                        let tps = if elapsed > 0.0 {
                            token_count as f32 / elapsed
                        } else {
                            0.0
                        };
                        let _ = tx
                            .send(StreamChunk {
                                text: reasoning.to_string(),
                                done: false,
                                is_reasoning: true,
                                stats: Some(InferenceStats {
                                    tokens_generated: token_count,
                                    tokens_prompt: 0,
                                    time_total_ms: (elapsed * 1000.0) as u64,
                                    tokens_per_second: tps,
                                }),
                            })
                            .await;
                    }

                    // Send actual content tokens
                    if !content.is_empty() {
                        full_text.push_str(content);
                        token_count += 1;
                        let elapsed = start_time.elapsed().as_secs_f32();
                        let tps = if elapsed > 0.0 {
                            token_count as f32 / elapsed
                        } else {
                            0.0
                        };
                        let _ = tx
                            .send(StreamChunk {
                                text: content.to_string(),
                                done: false,
                                is_reasoning: false,
                                stats: Some(InferenceStats {
                                    tokens_generated: token_count,
                                    tokens_prompt: 0,
                                    time_total_ms: (elapsed * 1000.0) as u64,
                                    tokens_per_second: tps,
                                }),
                            })
                            .await;
                    }

                    // Check finish_reason
                    let finish = json
                        .get("choices")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("finish_reason"))
                        .and_then(|v| v.as_str());
                    if let Some(reason) = finish {
                        if !reason.is_empty() && reason != "null" {
                            let usage = json.get("usage");
                            let elapsed = start_time.elapsed().as_secs_f32();
                            // Try server-reported tps first, fallback to our calculation
                            let tps = usage
                                .and_then(|u| u.get("timings"))
                                .and_then(|t| t.get("tokens_per_second"))
                                .and_then(|v| v.as_f64())
                                .map(|v| v as f32)
                                .unwrap_or_else(|| {
                                    if elapsed > 0.0 {
                                        token_count as f32 / elapsed
                                    } else {
                                        0.0
                                    }
                                });
                            let tokens_generated = usage
                                .and_then(|u| u.get("completion_tokens"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(token_count as u64)
                                as u32;
                            let tokens_prompt = usage
                                .and_then(|u| u.get("prompt_tokens"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                as u32;
                            let _ = tx
                                .send(StreamChunk {
                                    text: String::new(),
                                    done: true,
                                    is_reasoning: false,
                                    stats: Some(InferenceStats {
                                        tokens_generated,
                                        tokens_prompt,
                                        time_total_ms: (elapsed * 1000.0) as u64,
                                        tokens_per_second: tps,
                                    }),
                                })
                                .await;
                            return Ok(());
                        }
                    }
                }
            }
        }

        let elapsed = start_time.elapsed().as_secs_f32();
        let tps = if elapsed > 0.0 {
            token_count as f32 / elapsed
        } else {
            0.0
        };
        let _ = tx
            .send(StreamChunk {
                text: String::new(),
                done: true,
                is_reasoning: false,
                stats: Some(InferenceStats {
                    tokens_generated: token_count,
                    tokens_prompt: 0,
                    time_total_ms: (elapsed * 1000.0) as u64,
                    tokens_per_second: tps,
                }),
            })
            .await;
        Ok(())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        if !self.check_server_alive().await {
            error!("llama-server is not responding — completion impossible");
            return Err(AthenasError::Backend(
                "llama-server is not responding. Please restart the server.".to_string(),
            ));
        }

        let mut body = serde_json::json!({
            "prompt": request.prompt,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "n_predict": request.max_tokens.unwrap_or(2048),
            "stream": false,
            "stop": request.stop.as_deref().unwrap_or(&[]),
        });

        if let Some(ref grammar) = request.grammar {
            body["grammar"] = serde_json::Value::String(grammar.clone());
        }
        if let Some(ref rf) = request.response_format {
            body["response_format"] = rf.clone();
        }

        let url = format!("{}/completion", self.server_url());
        tracing::info!("Sending completion request to llama-server at {}", url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                error!("llama-server completion request failed: {}", e);
                AthenasError::Backend(format!("Request failed: {}", e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            error!("llama-server returned {} for completion: {}", status, text);
            return Err(AthenasError::Backend(format!(
                "llama-server returned {}: {}",
                status, text
            )));
        }

        let result: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AthenasError::Backend(format!("Invalid response: {}", e)))?;

        let content = result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let tokens_decoded = result
            .get("tokens_decoded")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let tokens_predicted = result
            .get("tokens_predicted")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let timings = result.get("timings");
        let tps = timings
            .and_then(|t| t.get("tokens_per_second"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let time_ms = timings
            .and_then(|t| t.get("predicted_ms"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        Ok(CompletionResponse {
            model: self.model_name.clone(),
            text: content,
            stats: InferenceStats {
                tokens_generated: tokens_decoded,
                tokens_prompt: tokens_predicted,
                time_total_ms: time_ms,
                tokens_per_second: tps,
            },
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        let chat_req = ChatRequest {
            model: request.model.clone(),
            messages: vec![ChatMessage::user(&request.prompt)],
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            stream: true,
            stop: request.stop.clone(),
            grammar: request.grammar.clone(),
            response_format: request.response_format.clone(),
            ..Default::default()
        };
        self.chat_stream(chat_req, tx).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        let inputs = request.input.as_vec();
        let body = serde_json::json!({
            "model": self.model_name,
            "input": inputs,
        });

        let url = format!("{}/v1/embeddings", self.server_url());
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("Embeddings request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!(
                "llama-server embeddings returned {}: {}",
                status, text
            )));
        }

        let result: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AthenasError::Backend(format!("Invalid embeddings response: {}", e)))?;

        let data_arr = result
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| {
                AthenasError::Backend("Missing 'data' in embeddings response".to_string())
            })?;

        let data: Vec<EmbeddingData> = data_arr
            .iter()
            .map(|d| {
                let embedding = d
                    .get("embedding")
                    .and_then(|e| e.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                let index = d.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                EmbeddingData {
                    object: "embedding".to_string(),
                    embedding,
                    index,
                }
            })
            .collect();

        let usage = result.get("usage");
        let prompt_tokens = usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let total_tokens = usage
            .and_then(|u| u.get("total_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(prompt_tokens as u64) as u32;

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data,
            model: self.model_name.clone(),
            usage: EmbeddingUsage {
                prompt_tokens,
                total_tokens,
            },
        })
    }

    fn model_info(&self) -> Option<ModelInfo> {
        if self.loaded {
            Some(ModelInfo {
                name: self.model_name.clone(),
                context_size: self.context_size,
                gpu_layers: self.gpu_layers,
                backend_name: "llama.cpp".to_string(),
            })
        } else {
            None
        }
    }

    /// Quick health check — pings the llama-server's /health endpoint
    /// with a short timeout. Returns false if the server is unresponsive.
    async fn health_check(&self) -> Result<bool> {
        if !self.loaded || self.server_port == 0 {
            return Ok(false);
        }
        let url = format!("http://127.0.0.1:{}/health", self.server_port);
        let result = self
            .client
            .get(&url)
            .timeout(tokio::time::Duration::from_secs(3))
            .send()
            .await;
        match result {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    fn boxed_clone(&self) -> Box<dyn Backend> {
        Box::new(LlamaCppBackend {
            hardware: self.hardware.clone(),
            loaded: self.loaded,
            model_path: self.model_path.clone(),
            model_name: self.model_name.clone(),
            context_size: self.context_size,
            gpu_layers: self.gpu_layers,
            server_handle: None, // Child is not Clone; not needed for streaming
            server_port: self.server_port,
            client: self.client.clone(),
            skip_reasoning_flag: self.skip_reasoning_flag,
            reasoning_enabled: self.reasoning_enabled,
            watchdog_handle: None, // Watchdog is not cloned
            watchdog_stop: self.watchdog_stop.clone(),
        })
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        // Stop the watchdog
        self.watchdog_stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
        if let Some(mut child) = self.server_handle.take() {
            let _ = child.start_kill();
        }
    }
}

/// Auto-detect a multimodal projector (mmproj) file in the same directory as the model.
/// Searches for files containing "mmproj", "vision", or "projector" in the name
/// with common extensions (.gguf, .bin, .safetensors).
fn auto_detect_mmproj(model_path: &str) -> Option<String> {
    let path = std::path::Path::new(model_path);
    let dir = path.parent()?;

    let entries = std::fs::read_dir(dir).ok()?;

    let model_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Patterns that indicate a multimodal projector file
    let mmproj_patterns = ["mmproj", "vision", "projector"];

    let mut candidates: Vec<(String, u64, usize)> = Vec::new();
    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_file() {
            if let Some(name) = entry_path.file_name().and_then(|n| n.to_str()) {
                let lower = name.to_lowercase();

                // Skip the model file itself
                if lower == model_name {
                    continue;
                }

                // Check for mmproj patterns
                for (idx, pattern) in mmproj_patterns.iter().enumerate() {
                    if lower.contains(pattern) {
                        // Must be a supported file extension
                        let is_valid_ext = lower.ends_with(".gguf")
                            || lower.ends_with(".bin")
                            || lower.ends_with(".safetensors");
                        if !is_valid_ext {
                            continue;
                        }
                        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                        // Prefer "mmproj" matches (idx=0) over "vision" (idx=1) over "projector" (idx=2)
                        candidates.push((entry_path.to_string_lossy().to_string(), size, idx));
                        break;
                    }
                }
            }
        }
    }

    if candidates.is_empty() {
        tracing::debug!("No mmproj file found in {}", dir.display());
        return None;
    }

    // Sort by pattern priority (mmproj first), then by size (largest first)
    candidates.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.1.cmp(&a.1)));

    let result = candidates.first().map(|(p, _, _)| p.clone());
    if let Some(ref p) = result {
        info!("Auto-detected mmproj: {}", p);
    }
    result
}

fn find_free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| {
            let addr = listener.local_addr()?;
            Ok(addr.port())
        })
        .unwrap_or(9090)
}
