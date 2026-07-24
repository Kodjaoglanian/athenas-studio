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
    // SAFETY: kill(pid, 0) doesn't send a signal — it just checks existence.
    // On Unix, this is safe. On Windows, we'd need a different approach.
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        // Fallback for non-Unix: assume alive if state file exists
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
    context_size: u32,
    max_concurrent: Option<u32>,
    rate_limit: Option<u32>,
    timeout_secs: Option<u64>,
    max_body_size_mb: Option<u32>,
) -> Result<ServerState, String> {
    let binary = athenas_binary().ok_or("Cannot find athenas binary")?;

    let mut cmd = std::process::Command::new(&binary);
    cmd.arg("serve")
        .arg(model)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--backend")
        .arg(backend)
        .arg("--gpu-layers")
        .arg(gpu_layers.to_string())
        .arg("--context-size")
        .arg(context_size.to_string());

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

    // Detach: redirect stdin to /dev/null, stdout to log file, stderr to log file
    let log_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".athenas");
    let _ = std::fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("server.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open server log: {}", e))?;

    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(
            log_file.try_clone().map_err(|e| e.to_string())?,
        ))
        .stderr(Stdio::from(log_file));

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
    #[cfg(not(unix))]
    {
        // On non-Unix, try to kill via the OS
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .spawn();
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
