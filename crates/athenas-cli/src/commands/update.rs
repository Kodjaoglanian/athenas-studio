use anyhow::{Context, Result};
use std::env;
use std::process::Command;

const REPO: &str = "Kodjaoglanian/athenas-studio";
const INSTALL_URL: &str =
    "https://github.com/Kodjaoglanian/athenas-studio/releases/latest/download/install.sh";

fn print_banner() {
    let cyan = "\x1b[0;36m";
    let bold = "\x1b[1m";
    let nc = "\x1b[0m";
    println!();
    println!(
        "{}{}    ___   __   ____  _   _ _____ _____ ____     {}",
        cyan, bold, nc
    );
    println!(
        "{}{}   / _ \\ / /_ / ___|| | |  ___|_   _|  _ \\    {}",
        cyan, bold, nc
    );
    println!(
        "{}{}  / /_\\_/ __|\\___ \\| |_| | |_    | | | |_) |   {}",
        cyan, bold, nc
    );
    println!(
        "{}{} / /_\\  \\_| |____) |  _  |  _|   | | |  _ <    {}",
        cyan, bold, nc
    );
    println!(
        "{}{} \\____|\\__|____/ |_| |_|_|     |_| |_| \\_\\   {}",
        cyan, bold, nc
    );
    println!("{}{}        Studio — Local LLM Inference{}", cyan, bold, nc);
    println!();
}

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
    print_banner();

    let current_version = get_current_version();
    info(&format!("Current version: {}", current_version));

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
