use std::path::PathBuf;
use tracing::{info, warn};

use athenas_core::{AthenasError, Result};

const LLAMA_CPP_REPO: &str = "ggml-org/llama.cpp";

/// Detect the platform-appropriate asset name for llama.cpp releases.
fn platform_asset_name() -> Option<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("linux", "x86_64") => Some("bin-ubuntu-x64.tar.gz".to_string()),
        ("linux", "aarch64") => Some("bin-ubuntu-arm64.tar.gz".to_string()),
        ("macos", "aarch64") => Some("bin-macos-arm64.tar.gz".to_string()),
        ("macos", "x86_64") => Some("bin-macos-x64.tar.gz".to_string()),
        ("windows", "x86_64") => Some("bin-win-cpu-x64.zip".to_string()),
        ("windows", "aarch64") => Some("bin-win-cpu-arm64.zip".to_string()),
        _ => {
            warn!("No prebuilt llama-server for os={} arch={}", os, arch);
            None
        }
    }
}

/// Query GitHub API for the latest llama.cpp release tag.
async fn get_latest_release_tag() -> Result<String> {
    let url = format!(
        "https://api.github.com/repos/{}/releases/latest",
        LLAMA_CPP_REPO
    );

    let client = reqwest::Client::builder()
        .user_agent("athenas-studio")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| AthenasError::Backend(format!("Failed to create HTTP client: {}", e)))?;

    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| AthenasError::Backend(format!("GitHub API request failed: {}", e)))?;

    if !resp.status().is_success() {
        return Err(AthenasError::Backend(format!(
            "GitHub API returned {}",
            resp.status()
        )));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| AthenasError::Backend(format!("Failed to parse GitHub response: {}", e)))?;

    let tag = json["tag_name"]
        .as_str()
        .ok_or_else(|| AthenasError::Backend("No tag_name in GitHub response".into()))?;

    Ok(tag.to_string())
}

/// Get the athenas bin directory (~/.athenas/bin).
fn athenas_bin_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| AthenasError::Backend("Cannot determine home directory".into()))?;

    let bin_dir = PathBuf::from(home).join(".athenas").join("bin");
    std::fs::create_dir_all(&bin_dir)
        .map_err(|e| AthenasError::Backend(format!("Failed to create bin dir: {}", e)))?;
    Ok(bin_dir)
}

/// Download a file from a URL and return the bytes.
async fn download_file(url: &str) -> Result<Vec<u8>> {
    info!("Downloading {}", url);

    let client = reqwest::Client::builder()
        .user_agent("athenas-studio")
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| AthenasError::Backend(format!("HTTP client error: {}", e)))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| AthenasError::Backend(format!("Download failed: {}", e)))?;

    if !resp.status().is_success() {
        return Err(AthenasError::Backend(format!(
            "Download failed with status {}",
            resp.status()
        )));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| AthenasError::Backend(format!("Failed to read response: {}", e)))?;

    Ok(bytes.to_vec())
}

/// Extract all files from a tar.gz archive into bin_dir.
/// Returns the path to llama-server.
fn extract_tar_gz(data: &[u8], bin_dir: &std::path::Path) -> Result<PathBuf> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let decoder = GzDecoder::new(data);
    let mut archive = Archive::new(decoder);

    let server_name = if std::env::consts::OS == "windows" {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let mut server_path = None;

    for entry in archive
        .entries()
        .map_err(|e| AthenasError::Backend(format!("Failed to read tar entries: {}", e)))?
    {
        let mut entry =
            entry.map_err(|e| AthenasError::Backend(format!("Failed to read tar entry: {}", e)))?;

        let file_name = entry
            .path()
            .map_err(|e| AthenasError::Backend(format!("Failed to get entry path: {}", e)))?
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        if file_name.is_empty() || file_name == "." || file_name == ".." {
            continue;
        }

        // Extract all files flat into bin_dir (flatten directory structure)
        let dest = bin_dir.join(&file_name);

        // Preserve permissions (makes executables executable)
        #[cfg(unix)]
        {
            entry.set_preserve_permissions(true);
        }

        entry.unpack(&dest).map_err(|e| {
            AthenasError::Backend(format!("Failed to extract {}: {}", file_name, e))
        })?;

        if file_name == server_name {
            server_path = Some(dest);
        }
    }

    server_path.ok_or_else(|| AthenasError::Backend("llama-server not found in archive".into()))
}

/// Extract all files from a zip archive into bin_dir.
/// Returns the path to llama-server.exe.
fn extract_zip(data: &[u8], bin_dir: &std::path::Path) -> Result<PathBuf> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| AthenasError::Backend(format!("Failed to open zip: {}", e)))?;

    let server_name = "llama-server.exe";
    let mut server_path = None;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| AthenasError::Backend(format!("Failed to read zip entry: {}", e)))?;

        let name = file.name().to_string();

        // Skip directories
        if name.ends_with('/') {
            continue;
        }

        // Flatten: just take the file name (strip any directory prefix)
        let file_name = std::path::Path::new(&name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        if file_name.is_empty() {
            continue;
        }

        let dest = bin_dir.join(file_name);
        let mut out = std::fs::File::create(&dest)
            .map_err(|e| AthenasError::Backend(format!("Failed to create {}: {}", file_name, e)))?;
        std::io::copy(&mut file, &mut out)
            .map_err(|e| AthenasError::Backend(format!("Failed to write {}: {}", file_name, e)))?;

        if file_name == server_name {
            server_path = Some(dest);
        }
    }

    server_path.ok_or_else(|| AthenasError::Backend("llama-server.exe not found in zip".into()))
}

/// Auto-download and install llama-server to ~/.athenas/bin/
pub async fn ensure_llama_server() -> Result<PathBuf> {
    let bin_dir = athenas_bin_dir()?;

    let server_name = if std::env::consts::OS == "windows" {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let server_path = bin_dir.join(server_name);

    // Already installed?
    if server_path.exists() {
        return Ok(server_path);
    }

    info!("llama-server not found, auto-downloading...");

    let asset_suffix = platform_asset_name().ok_or_else(|| {
        AthenasError::Backend(format!(
            "No prebuilt llama-server available for {} {}. Please install llama.cpp manually.",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
    })?;

    let tag = get_latest_release_tag().await?;
    info!("Latest llama.cpp release: {}", tag);

    let asset_name = format!("llama-{}-{}", tag, asset_suffix);
    let download_url = format!(
        "https://github.com/{}/releases/download/{}/{}",
        LLAMA_CPP_REPO, tag, asset_name
    );

    let data = download_file(&download_url).await?;
    info!(
        "Downloaded {} ({} MB), extracting...",
        asset_name,
        data.len() / (1024 * 1024)
    );

    let is_zip = asset_suffix.ends_with(".zip");
    let extracted_path = if is_zip {
        extract_zip(&data, &bin_dir)?
    } else {
        extract_tar_gz(&data, &bin_dir)?
    };

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&extracted_path)
            .map_err(|e| AthenasError::Backend(format!("stat error: {}", e)))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&extracted_path, perms)
            .map_err(|e| AthenasError::Backend(format!("chmod error: {}", e)))?;
    }

    // Verify the binary actually works
    info!("Verifying llama-server binary...");
    let mut verify_cmd = tokio::process::Command::new(&extracted_path);
    verify_cmd
        .arg("--version")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    // Set LD_LIBRARY_PATH so it finds shared libs in the same dir
    if let Some(parent) = extracted_path.parent() {
        let lib_path = parent.to_string_lossy().to_string();
        let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let new_ld_path = if existing.is_empty() {
            lib_path
        } else {
            format!("{}:{}", lib_path, existing)
        };
        verify_cmd.env("LD_LIBRARY_PATH", new_ld_path);
    }

    let verify = verify_cmd.output().await;

    match verify {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            info!(
                "llama-server verified: {}",
                version.lines().next().unwrap_or("ok")
            );
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(AthenasError::Backend(format!(
                "Downloaded llama-server failed to run (exit code: {:?}).\n\
                 stdout: {}\n\
                 stderr: {}\n\
                 This usually means missing shared libraries.\n\
                 Try: ldd {}\n\
                 On Ubuntu/Debian: apt install -y libgomp1",
                output.status.code(),
                stdout,
                stderr,
                extracted_path.display()
            )));
        }
        Err(e) => {
            return Err(AthenasError::Backend(format!(
                "Cannot execute downloaded llama-server: {}\n\
                 Path: {}\n\
                 Try: ldd {} to check for missing libraries",
                e,
                extracted_path.display(),
                extracted_path.display()
            )));
        }
    }

    info!("llama-server installed to {}", extracted_path.display());
    Ok(extracted_path)
}
