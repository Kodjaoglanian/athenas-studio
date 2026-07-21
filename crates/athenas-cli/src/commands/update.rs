use anyhow::{Context, Result};
use std::env;
use std::process::Command;

const REPO: &str = "Kodjaoglanian/athenas-studio";
const INSTALL_URL: &str =
    "https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.sh";

pub async fn run() -> Result<()> {
    println!();
    println!(" ░▒▓██████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░ ░▒▓███████▓▒░       ░▒▓███████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓██████▓▒░  ");
    println!("░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ");
    println!("░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ");
    println!("░▒▓████████▓▒░ ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░░▒▓██████▓▒░        ░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ");
    println!("░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░             ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ");
    println!("░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░             ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ");
    println!("░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░       ░▒▓███████▓▒░   ░▒▓█▓▒░    ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓██████▓▒░  ");
    println!();

    let current_version = get_current_version();
    println!("[info] Current version: {}", current_version);

    println!("[info] Checking for latest release...");
    let latest_version = get_latest_version().await?;
    println!("[info] Latest version:  {}", latest_version);

    if current_version == latest_version {
        println!();
        println!("[ok] You're already up to date!");
        return Ok(());
    }

    println!();
    println!(
        "[info] Updating from {} to {}...",
        current_version, latest_version
    );
    println!();

    let platform = env::consts::OS;
    match platform {
        "linux" | "macos" | "freebsd" | "openbsd" | "netbsd" => {
            run_install_script().await?;
        }
        "windows" => {
            println!("[info] On Windows, please run the following in PowerShell:");
            println!();
            println!(
                "  irm https://github.com/{}/releases/latest/download/install.ps1 | iex",
                REPO
            );
            println!();
        }
        _ => {
            println!(
                "[error] Unsupported platform: {}. Please download manually from:",
                platform
            );
            println!("  https://github.com/{}/releases/latest", REPO);
        }
    }

    Ok(())
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
    println!("[info] Downloading and running installer...");
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
    println!("[ok] Update complete! Run 'athenas --version' to verify.");
    Ok(())
}
