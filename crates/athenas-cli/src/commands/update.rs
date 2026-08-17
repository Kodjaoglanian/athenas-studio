use anyhow::{Context, Result};
use std::env;
use std::process::Command;

const REPO: &str = "Kodjaoglanian/athenas-studio";
const INSTALL_URL: &str =
    "https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.sh";

fn info(msg: &str) {
    let cyan = "\x1b[0;36m";
    let nc = "\x1b[0m";
    println!("  {}[info]{} {}", cyan, nc, msg);
}

fn success(msg: &str) {
    let green = "\x1b[0;32m";
    let nc = "\x1b[0m";
    println!("  {}[ok]{}   {}", green, nc, msg);
}

#[allow(dead_code)]
fn warn(msg: &str) {
    let yellow = "\x1b[1;33m";
    let nc = "\x1b[0m";
    println!("  {}[warn]{} {}", yellow, nc, msg);
}

fn error(msg: &str) {
    let red = "\x1b[0;31m";
    let nc = "\x1b[0m";
    println!("  {}[err]{}  {}", red, nc, msg);
}

pub async fn run() -> Result<()> {
    let current_version = get_current_version();
    info(&format!("Current version: {}", current_version));

    // Check if athenas or llama-server processes are running
    check_running_processes()?;

    info("Checking for latest release...");
    let latest_version = get_latest_version().await?;
    info(&format!("Latest version:  {}", latest_version));

    if current_version == latest_version {
        println!();
        success("You're already up to date!");
        println!();
        return Ok(());
    }

    println!();
    info(&format!(
        "Updating from {} to {}...",
        current_version, latest_version
    ));
    println!();

    let platform = env::consts::OS;
    match platform {
        "linux" | "macos" | "freebsd" | "openbsd" | "netbsd" => {
            run_install_script().await?;
        }
        "windows" => {
            info("On Windows, please run the following in PowerShell:");
            println!();
            println!(
                "  irm https://github.com/{}/releases/latest/download/install.ps1 | iex",
                REPO
            );
            println!();
        }
        _ => {
            error(&format!(
                "Unsupported platform: {}. Please download manually from:",
                platform
            ));
            println!("  https://github.com/{}/releases/latest", REPO);
        }
    }

    Ok(())
}

/// Check if athenas or llama-server processes are running.
/// On Windows, the installer cannot overwrite a running binary.
/// On Linux/macOS, the install script handles this, but we warn anyway.
fn check_running_processes() -> Result<()> {
    let found = find_running_processes();
    if !found.is_empty() {
        error("Cannot update while Athenas Studio or llama-server is running.");
        println!();
        println!("  The following processes are still active:");
        for (name, pid) in &found {
            println!("    {} (PID: {})", name, pid);
        }
        println!();
        println!("  Please stop the server and close Athenas Studio before updating:");
        if cfg!(target_os = "windows") {
            println!("    Stop-Process -Name athenas,llama-server -Force");
        } else {
            println!("    killall athenas llama-server");
        }
        println!();
        anyhow::bail!("Stop the running processes and try again.");
    }
    Ok(())
}

/// Find running athenas and llama-server processes.
fn find_running_processes() -> Vec<(String, u32)> {
    let mut found = Vec::new();

    #[cfg(target_os = "windows")]
    {
        // Use tasklist to find processes
        for name in &["athenas.exe", "llama-server.exe"] {
            if let Ok(output) = Command::new("tasklist")
                .args([
                    "/FI",
                    &format!("IMAGENAME eq {}", name),
                    "/FO",
                    "CSV",
                    "/NH",
                ])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    // CSV format: "Image Name","PID","Session Name","Session#","Mem Usage"
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 2 {
                        let pid: u32 = parts[1].trim_matches('"').parse().unwrap_or(0);
                        if pid > 0 {
                            found.push((name.to_string(), pid));
                        }
                    }
                }
            }
        }
    }

    #[cfg(unix)]
    {
        // Use pgrep to find processes
        for name in &["athenas", "llama-server"] {
            if let Ok(output) = Command::new("pgrep").arg("-x").arg(name).output() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    if let Ok(pid) = line.trim().parse::<u32>() {
                        found.push((name.to_string(), pid));
                    }
                }
            }
        }
    }

    found
}

fn get_current_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

async fn get_latest_version() -> Result<String> {
    let url = format!("https://api.github.com/repos/{}/releases/latest", REPO);
    let client = reqwest::Client::builder()
        .user_agent("athenas-studio-updater")
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp: serde_json::Value = client
        .get(&url)
        .send()
        .await?
        .json()
        .await
        .context("Failed to fetch latest release info")?;

    let tag = resp
        .get("tag_name")
        .and_then(|v| v.as_str())
        .context("No tag_name in release response")?;

    Ok(tag.trim_start_matches('v').to_string())
}

async fn run_install_script() -> Result<()> {
    info("Downloading and running installer...");
    println!();

    let status = Command::new("bash")
        .arg("-c")
        .arg(format!("curl -fsSL {} | bash", INSTALL_URL))
        .status()
        .context("Failed to run install script")?;

    if !status.success() {
        anyhow::bail!("Install script failed with exit code: {:?}", status.code());
    }

    println!();
    success("Update complete! Run 'athenas --version' to verify.");
    Ok(())
}
