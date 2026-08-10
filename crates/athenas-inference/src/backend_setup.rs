use std::path::PathBuf;
use tracing::{info, warn};

use athenas_core::{AthenasError, Result};

const LLAMA_CPP_REPO: &str = "ggml-org/llama.cpp";

/// Detect the platform-appropriate asset name for llama.cpp releases.
///
/// This function checks for GPU availability and selects the correct
/// GPU-accelerated binary. If no GPU is detected, it falls back to CPU.
///
/// Priority order:
/// - NVIDIA GPU: CUDA (Windows) / Vulkan (Linux — no CUDA prebuilt for Linux)
/// - AMD GPU: ROCm (if available) / Vulkan
/// - Intel GPU: SYCL (Linux) / OpenVINO (Windows)
/// - Apple Silicon: Metal (built into macOS binaries)
/// - No GPU: CPU-only binary
fn platform_asset_name() -> Option<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    // Detect GPU capabilities
    let has_nvidia = detect_nvidia();
    let has_amd = detect_amd();
    let has_vulkan = detect_vulkan_support();

    match (os, arch) {
        // Linux x86_64
        ("linux", "x86_64") => {
            // On Linux, always prefer Vulkan because:
            // 1. There is no CUDA prebuilt for Linux from llama.cpp
            // 2. Many AMD APUs (Barcelo, Renoir, etc.) have rocm-smi
            //    installed but don't actually support ROCm compute
            // 3. Vulkan works with NVIDIA, AMD, and Intel GPUs
            if has_vulkan {
                if has_nvidia {
                    info!("NVIDIA GPU detected, using Vulkan binary for GPU acceleration");
                } else if has_amd {
                    info!("AMD GPU detected, using Vulkan binary for GPU acceleration");
                } else {
                    info!("Vulkan support detected, using Vulkan binary for GPU acceleration");
                }
                return Some("bin-ubuntu-vulkan-x64.tar.gz".to_string());
            }
            // No Vulkan — try ROCm as fallback (AMD only)
            if has_amd && detect_rocm() {
                info!("AMD GPU detected with ROCm (no Vulkan), using ROCm binary");
                return Some("bin-ubuntu-rocm-7.2-x64.tar.gz".to_string());
            }
            // No Vulkan, no ROCm — try CUDA (custom build)
            if has_nvidia {
                warn!("NVIDIA GPU detected but no Vulkan — falling back to CPU binary");
            }
            info!("No GPU/Vulkan detected, using CPU-only binary");
            Some("bin-ubuntu-x64.tar.gz".to_string())
        }
        // Linux aarch64
        ("linux", "aarch64") => {
            if has_vulkan {
                info!("Vulkan support detected, using Vulkan binary for GPU acceleration");
                return Some("bin-ubuntu-vulkan-arm64.tar.gz".to_string());
            }
            Some("bin-ubuntu-arm64.tar.gz".to_string())
        }
        // macOS — Metal is built into all macOS binaries
        ("macos", "aarch64") => Some("bin-macos-arm64.tar.gz".to_string()),
        ("macos", "x86_64") => Some("bin-macos-x64.tar.gz".to_string()),
        // Windows x86_64
        ("windows", "x86_64") => {
            if has_nvidia {
                // CUDA 12.4 is the most widely compatible
                info!("NVIDIA GPU detected, using CUDA 12.4 binary for GPU acceleration");
                return Some("bin-win-cuda-12.4-x64.zip".to_string());
            }
            if has_amd {
                info!("AMD GPU detected, using HIP binary for GPU acceleration");
                return Some("bin-win-hip-radeon-x64.zip".to_string());
            }
            if has_vulkan {
                info!("Vulkan support detected, using Vulkan binary for GPU acceleration");
                return Some("bin-win-vulkan-x64.zip".to_string());
            }
            info!("No GPU detected, using CPU-only binary");
            Some("bin-win-cpu-x64.zip".to_string())
        }
        ("windows", "aarch64") => Some("bin-win-cpu-arm64.zip".to_string()),
        _ => {
            warn!("No prebuilt llama-server for os={} arch={}", os, arch);
            None
        }
    }
}

/// Check if an NVIDIA GPU is present by running nvidia-smi.
fn detect_nvidia() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false)
}

/// Check if an AMD GPU is present by running rocm-smi.
fn detect_amd() -> bool {
    std::process::Command::new("rocm-smi")
        .arg("--showproductname")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if ROCm is installed (rocm-smi works).
fn detect_rocm() -> bool {
    std::process::Command::new("rocm-smi")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Vulkan is available by running vulkaninfo or checking for
/// libvulkan.so on Linux.
fn detect_vulkan_support() -> bool {
    if cfg!(target_os = "linux") {
        // Method 1: try vulkaninfo
        if std::process::Command::new("vulkaninfo")
            .arg("--summary")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return true;
        }
        // Method 2: check for libvulkan.so in common paths
        let lib_paths = [
            "/usr/lib/x86_64-linux-gnu/libvulkan.so",
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/usr/lib/libvulkan.so",
            "/usr/lib/libvulkan.so.1",
            "/usr/local/lib/libvulkan.so",
            "/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/lib/libvulkan.so.1",
        ];
        for path in &lib_paths {
            if std::path::Path::new(path).exists() {
                return true;
            }
        }
        // Method 3: check if NVIDIA driver is installed (provides Vulkan)
        if std::path::Path::new("/usr/lib/x86_64-linux-gnu/libGL.so.1").exists()
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libnvidia-glcore.so").exists()
            || std::path::Path::new("/proc/driver/nvidia").exists()
        {
            // NVIDIA driver is installed — Vulkan should work even if
            // vulkaninfo isn't installed
            return true;
        }
        false
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("vulkaninfo")
            .arg("--summary")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    } else {
        false
    }
}

/// Query GitHub API for the latest llama.cpp release tag.
/// If `required_asset` is provided, searches recent releases for one
/// that contains an asset with the given suffix (e.g. "bin-ubuntu-vulkan-x64.tar.gz").
/// This is needed because the latest release sometimes doesn't have all assets.
async fn get_latest_release_tag(required_asset: Option<&str>) -> Result<String> {
    let client = reqwest::Client::builder()
        .user_agent("athenas-studio")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| AthenasError::Backend(format!("Failed to create HTTP client: {}", e)))?;

    // If no specific asset is needed, just get the latest release
    if required_asset.is_none() {
        let url = format!(
            "https://api.github.com/repos/{}/releases/latest",
            LLAMA_CPP_REPO
        );
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

        let json: serde_json::Value = resp.json().await.map_err(|e| {
            AthenasError::Backend(format!("Failed to parse GitHub response: {}", e))
        })?;

        let tag = json["tag_name"]
            .as_str()
            .ok_or_else(|| AthenasError::Backend("No tag_name in GitHub response".into()))?;

        return Ok(tag.to_string());
    }

    // Search recent releases for one that has the required asset
    let required = required_asset.unwrap();
    let url = format!(
        "https://api.github.com/repos/{}/releases?per_page=10",
        LLAMA_CPP_REPO
    );

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

    let releases: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| AthenasError::Backend(format!("Failed to parse GitHub response: {}", e)))?;

    let releases_arr = releases
        .as_array()
        .ok_or_else(|| AthenasError::Backend("Expected array of releases".into()))?;

    for release in releases_arr {
        let tag = release["tag_name"]
            .as_str()
            .ok_or_else(|| AthenasError::Backend("No tag_name in release".into()))?;

        if let Some(assets) = release["assets"].as_array() {
            for asset in assets {
                if let Some(name) = asset["name"].as_str() {
                    if name.contains(required) {
                        info!("Found release {} with asset matching '{}'", tag, required);
                        return Ok(tag.to_string());
                    }
                }
            }
        }
    }

    Err(AthenasError::Backend(format!(
        "No release found with asset containing '{}'. \
         The llama.cpp releases may have changed format.",
        required
    )))
}

/// Get the athenas bin directory (~/.athenas/bin).
fn athenas_bin_dir() -> Result<PathBuf> {
    let home = dirs::home_dir()
        .ok_or_else(|| AthenasError::Backend("Cannot determine home directory".into()))?;

    let bin_dir = home.join(".athenas").join("bin");
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
///
/// If `force_variant` is Some, re-downloads with the specified variant
/// even if a binary already exists. This is used when the user changes
/// GPU settings and needs a GPU-accelerated binary.
pub async fn ensure_llama_server() -> Result<PathBuf> {
    ensure_llama_server_with_variant(None).await
}

/// Force re-download of llama-server with a specific GPU variant.
/// This removes the existing binary and downloads the correct one.
pub async fn force_redownload_llama_server() -> Result<PathBuf> {
    ensure_llama_server_with_variant(Some(true)).await
}

async fn ensure_llama_server_with_variant(force_redownload: Option<bool>) -> Result<PathBuf> {
    let bin_dir = athenas_bin_dir()?;

    let server_name = if std::env::consts::OS == "windows" {
        "llama-server.exe"
    } else {
        "llama-server"
    };

    let server_path = bin_dir.join(server_name);
    let variant_marker = bin_dir.join(".llama-server-variant");

    // Determine the desired asset (GPU-aware)
    let desired_asset_suffix = platform_asset_name().ok_or_else(|| {
        AthenasError::Backend(format!(
            "No prebuilt llama-server available for {} {}. Please install llama.cpp manually.",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
    })?;

    // Check if we need to re-download:
    // 1. Binary doesn't exist → download
    // 2. Force re-download requested → download
    // 3. Binary exists but variant doesn't match desired (CPU vs GPU) → re-download
    let needs_download = if force_redownload == Some(true) {
        info!("Force re-download requested, replacing llama-server...");
        cleanup_bin_dir(&bin_dir, &server_path);
        true
    } else if !server_path.exists() {
        true
    } else {
        // Check if the variant matches what we need
        let current_variant = std::fs::read_to_string(&variant_marker).unwrap_or_default();
        let desired_variant = &desired_asset_suffix;
        if current_variant.trim() != desired_variant.trim() {
            info!(
                "llama-server variant mismatch: have '{}', need '{}' — re-downloading with GPU support...",
                current_variant.trim(),
                desired_variant.trim()
            );
            cleanup_bin_dir(&bin_dir, &server_path);
            true
        } else {
            // Check for shared libs on Linux/macOS
            let lib_ok = if std::env::consts::OS == "linux" {
                std::fs::read_dir(&bin_dir)
                    .map(|entries| {
                        entries
                            .filter_map(|e| e.ok())
                            .any(|e| e.file_name().to_string_lossy().starts_with("libllama"))
                    })
                    .unwrap_or(false)
            } else if std::env::consts::OS == "macos" {
                std::fs::read_dir(&bin_dir)
                    .map(|entries| {
                        entries
                            .filter_map(|e| e.ok())
                            .any(|e| e.file_name().to_string_lossy().ends_with(".dylib"))
                    })
                    .unwrap_or(false)
            } else {
                true
            };

            if !lib_ok {
                info!("llama-server exists but shared libs missing, re-downloading...");
                cleanup_bin_dir(&bin_dir, &server_path);
                true
            } else {
                false
            }
        }
    };

    if !needs_download {
        return Ok(server_path);
    }

    info!("llama-server not found or needs update, auto-downloading...");

    // Search for a release that has the asset we need.
    // The latest release sometimes doesn't have all platform assets.
    let tag = get_latest_release_tag(Some(&desired_asset_suffix)).await?;
    info!("Using llama.cpp release: {}", tag);

    let asset_name = format!("llama-{}-{}", tag, desired_asset_suffix);
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

    let is_zip = desired_asset_suffix.ends_with(".zip");
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

    // On Windows, prevent a console window from popping up during verification
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        verify_cmd.creation_flags(CREATE_NO_WINDOW);
    }

    // Set LD_LIBRARY_PATH (Unix) so it finds shared libs in the same dir.
    // On Windows, DLLs are found via the executable's directory automatically.
    #[cfg(unix)]
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

    // Write variant marker so we know which GPU variant was installed
    let _ = std::fs::write(&variant_marker, &desired_asset_suffix);

    info!("llama-server installed to {}", extracted_path.display());
    Ok(extracted_path)
}

/// Remove the llama-server binary and shared libs from the bin directory.
fn cleanup_bin_dir(bin_dir: &std::path::Path, server_path: &std::path::Path) {
    let _ = std::fs::remove_file(server_path);
    if let Ok(entries) = std::fs::read_dir(bin_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("libllama") || name.starts_with("libggml") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
}
