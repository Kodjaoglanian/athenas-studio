use std::path::PathBuf;
use std::process::Stdio;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Persistent state of a running server — saved to disk so the TUI can
/// re-attach to a server that was started in a previous session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerState {
    pub pid: u32,
    pub host: String,
    pub port: u16,
    pub model: String,
    pub backend: String,
    pub started_at: String,
}

impl ServerState {
    fn state_file() -> PathBuf {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home.join(".athenas").join("server_state.json")
    }

    /// Save server state to disk.
    pub fn save(&self) {
        let path = Self::state_file();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            if let Err(e) = std::fs::write(&path, json) {
                warn!("Failed to save server state: {}", e);
            }
        }
    }

    /// Load server state from disk.
    pub fn load() -> Option<ServerState> {
        let path = Self::state_file();
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    }

    /// Remove the state file (called when server is stopped).
    pub fn clear() {
        let path = Self::state_file();
        let _ = std::fs::remove_file(&path);
    }
}

/// Check if a process with the given PID is still running.
pub fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: kill(pid, 0) doesn't send a signal — it just checks existence.
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };

        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle == 0 {
                // Can't open — process doesn't exist or no access
                return false;
            }
            let mut exit_code: u32 = 0;
            let ok = GetExitCodeProcess(handle, &mut exit_code);
            CloseHandle(handle);
            // ok is non-zero on success; STILL_ACTIVE means the process is running
            ok != 0 && exit_code == STILL_ACTIVE
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        // Unknown platform: assume alive
        true
    }
}

/// Check if the server is responding at the given URL.
async fn check_server_health(host: &str, port: u16) -> bool {
    let url = format!("http://{}:{}/v1/health", host, port);
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client.get(&url).send().await.is_ok()
}

/// Get the path to the athenas binary (self-executable).
fn athenas_binary() -> Option<PathBuf> {
    std::env::current_exe().ok()
}

/// Start the server as a detached child process.
///
/// Returns the PID and server state on success.
#[allow(clippy::too_many_arguments)]
pub fn start_detached(
    model: &str,
    host: &str,
    port: u16,
    backend: &str,
    gpu_layers: i32,
    gpu_runtime: &str,
    gpu_device: Option<u32>,
    context_size: u32,
    max_concurrent: Option<u32>,
    rate_limit: Option<u32>,
    timeout_secs: Option<u64>,
    max_body_size_mb: Option<u32>,
) -> Result<ServerState, String> {
    let binary = athenas_binary().ok_or("Cannot find athenas binary")?;

    let mut cmd = std::process::Command::new(&binary);
    cmd.arg("serve")
        .arg("--verbose")
        .arg(model)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--backend")
        .arg(backend)
        .arg("--gpu-layers")
        .arg(gpu_layers.to_string())
        .arg("--gpu-runtime")
        .arg(gpu_runtime)
        .arg("--context-size")
        .arg(context_size.to_string());

    if let Some(device) = gpu_device {
        cmd.arg("--gpu-device").arg(device.to_string());
    }

    if let Some(mc) = max_concurrent {
        cmd.arg("--max-concurrent").arg(mc.to_string());
    }
    if let Some(rl) = rate_limit {
        cmd.arg("--rate-limit").arg(rl.to_string());
    }
    if let Some(t) = timeout_secs {
        cmd.arg("--timeout").arg(t.to_string());
    }
    if let Some(bs) = max_body_size_mb {
        cmd.arg("--max-body-size").arg(bs.to_string());
    }

    // Detach: redirect stdin to /dev/null, stdout to /dev/null, stderr to
    // log file. Only stderr is captured because tracing_subscriber::fmt()
    // writes to stderr. Redirecting stdout (println! output: banner,
    // "Loading model:", etc.) to the same file as stderr causes
    // interleaving corruption — the two file descriptors write
    // independently and can overwrite each other mid-line.
    let log_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".athenas");
    let _ = std::fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("server.log");

    // Rotate: if the existing log file is > 10 MB, move it to server.log.old
    // This preserves the previous session's logs for debugging.
    if let Ok(metadata) = std::fs::metadata(&log_path) {
        if metadata.len() > 10 * 1024 * 1024 {
            let _ = std::fs::rename(&log_path, log_dir.join("server.log.old"));
        }
    }

    // Append to the log file instead of truncating, so logs from multiple
    // sessions accumulate. The log tailer in the TUI handles file rotation
    // by detecting when the file size shrinks.
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open server log: {}", e))?;

    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file));

    // Set GPU runtime environment variables based on the chosen runtime
    // and device index. These are inherited by the child process and
    // ultimately by the llama-server subprocess.
    if let Some(device) = gpu_device {
        match gpu_runtime {
            "cuda" => {
                cmd.env("CUDA_VISIBLE_DEVICES", device.to_string());
            }
            "rocm" => {
                cmd.env("HIP_VISIBLE_DEVICES", device.to_string());
                cmd.env("ROCR_VISIBLE_DEVICES", device.to_string());
            }
            "vulkan" => {
                cmd.env("GGML_VULKAN_DEVICE", device.to_string());
            }
            _ => {}
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }

    #[cfg(windows)]
    {
        // CREATE_NO_WINDOW (0x08000000) prevents a console window from
        // popping up when the server is started from the TUI.
        // CREATE_NEW_PROCESS_GROUP (0x00000200) makes it independent
        // so Ctrl-C in the TUI doesn't kill the server.
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to start server: {}", e))?;
    let pid = child.id();

    // Don't wait — let it run in the background
    drop(child);

    let state = ServerState {
        pid,
        host: host.to_string(),
        port,
        model: model.to_string(),
        backend: backend.to_string(),
        started_at: chrono::Utc::now().to_rfc3339(),
    };
    state.save();

    info!("Server started as detached process (PID: {})", pid);
    Ok(state)
}

/// Stop the server by sending SIGTERM to the process group.
/// Uses negative PID to kill the entire process group (athenas serve + llama-server child).
pub fn stop_by_pid(pid: u32) -> Result<(), String> {
    if !is_process_alive(pid) {
        return Err("Server process is not running".to_string());
    }

    #[cfg(unix)]
    {
        unsafe {
            // Send SIGTERM to the entire process group (negative PID).
            // The detached process called setsid(), so it's a session leader
            // and killing -pid targets the whole group, including llama-server.
            if libc::kill(-(pid as i32), libc::SIGTERM) != 0 {
                // Fallback: try killing just the process
                if libc::kill(pid as i32, libc::SIGTERM) != 0 {
                    return Err(format!("Failed to send SIGTERM to PID {}", pid));
                }
            }
        }
    }
    #[cfg(windows)]
    {
        // /T = kill child processes (llama-server), /F = force
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status();
    }

    // Wait a moment for graceful shutdown
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Force kill if still alive
    if is_process_alive(pid) {
        #[cfg(unix)]
        {
            unsafe {
                // Kill the entire process group
                libc::kill(-(pid as i32), libc::SIGKILL);
                // Also try individual PID as fallback
                libc::kill(pid as i32, libc::SIGKILL);
            }
        }
        #[cfg(windows)]
        {
            // taskkill /F already forces, but try again in case
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .status();
        }
    }

    ServerState::clear();
    Ok(())
}

/// Check if a server is currently running (from saved state + health check).
pub async fn check_running() -> Option<ServerState> {
    let state = ServerState::load()?;

    // Check if the process is alive
    if !is_process_alive(state.pid) {
        info!(
            "Server state found but PID {} is not alive — cleaning up",
            state.pid
        );
        ServerState::clear();
        return None;
    }

    // Double-check with a health endpoint
    if check_server_health(&state.host, state.port).await {
        Some(state)
    } else {
        // Process is alive but not responding yet — could be still loading
        // Return the state anyway so the UI can show "loading"
        Some(state)
    }
}
