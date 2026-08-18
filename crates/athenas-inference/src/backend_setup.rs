use std::path::PathBuf;
use tracing::{info, warn};

use athenas_core::{AthenasError, Result};

const LLAMA_CPP_REPO: &str = "ggml-org/llama.cpp";

/// Detect the platform-appropriate asset name for llama.cpp releases.
///
/// This function checks for GPU availability and selects the correct
/// GPU-accelerated binary. If no GPU is detected, it falls back to CPU.
///
/// Priority order on Linux:
/// - Vulkan (works with NVIDIA, AMD, and Intel GPUs/APUs)
/// - ROCm (only for dedicated AMD Radeon RX/Pro/Instinct GPUs — APUs
///   like Radeon 780M/890M have rocm-smi but don't support ROCm compute)
/// - CPU-only fallback
///
/// Priority order on Windows:
/// - NVIDIA: CUDA
/// - AMD: HIP
/// - Vulkan fallback
/// - CPU-only
async fn platform_asset_name() -> Option<String> {
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
            // 2. Many AMD APUs (Renoir, Strix, etc.) have rocm-smi
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
            // No Vulkan libs installed. Check if we have an AMD GPU that
            // could benefit from Vulkan — if so, try to auto-install the
            // Vulkan libraries before falling back.
            if has_amd || has_nvidia {
                warn!(
                    "GPU detected but Vulkan libraries not found. \
                     Attempting to install Vulkan libraries..."
                );
                if try_install_vulkan_libs_pub().await {
                    info!("Vulkan libraries installed successfully, using Vulkan binary");
                    return Some("bin-ubuntu-vulkan-x64.tar.gz".to_string());
                }
                warn!(
                    "Failed to auto-install Vulkan libraries. \
                     Please install manually: apt install -y libvulkan1 mesa-vulkan-drivers"
                );
            }
            // No Vulkan — try ROCm only for dedicated AMD GPUs (not APUs)
            if has_amd && detect_rocm() && !is_amd_apu() {
                info!("Dedicated AMD GPU detected with ROCm (no Vulkan), using ROCm binary");
                return Some("bin-ubuntu-rocm-7.2-x64.tar.gz".to_string());
            }
            if has_amd && detect_rocm() && is_amd_apu() {
                warn!(
                    "AMD APU detected with rocm-smi, but ROCm compute is not supported \
                     on integrated GPUs. Falling back to CPU. \
                     Install Vulkan libraries for GPU acceleration: \
                     apt install -y libvulkan1 mesa-vulkan-drivers"
                );
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
            if has_amd && !is_amd_apu() {
                // Dedicated AMD GPU — use HIP binary
                info!("Dedicated AMD GPU detected, using HIP binary for GPU acceleration");
                return Some("bin-win-hip-radeon-x64.zip".to_string());
            }
            if has_amd && is_amd_apu() && has_vulkan {
                // AMD APU (integrated) — HIP doesn't work, use Vulkan
                info!("AMD APU detected, using Vulkan binary for GPU acceleration");
                return Some("bin-win-vulkan-x64.zip".to_string());
            }
            if has_amd && is_amd_apu() && !has_vulkan {
                // AMD APU without Vulkan — fall back to CPU
                warn!(
                    "AMD APU detected but no Vulkan support. \
                     HIP/ROCm is not supported on integrated GPUs. \
                     Falling back to CPU. Install Vulkan drivers for GPU acceleration."
                );
                return Some("bin-win-cpu-x64.zip".to_string());
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

/// Check if an NVIDIA GPU is present.
/// Uses nvidia-smi (works on both Linux and Windows when driver is installed).
/// Falls back to PowerShell Get-CimInstance on Windows if nvidia-smi is not in PATH.
fn detect_nvidia() -> bool {
    if std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false)
    {
        return true;
    }
    #[cfg(target_os = "windows")]
    {
        // Fallback: check via PowerShell for NVIDIA display adapters
        let ps_script =
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name";
        std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", ps_script])
            .output()
            .map(|o| {
                if !o.status.success() {
                    return false;
                }
                let stdout = String::from_utf8_lossy(&o.stdout).to_lowercase();
                stdout.contains("nvidia") || stdout.contains("geforce") || stdout.contains("quadro")
            })
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

/// Check if an AMD GPU is present.
/// On Linux: uses rocm-smi.
/// On Windows: uses PowerShell Get-CimInstance (WMIC is deprecated on Win11).
fn detect_amd() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("rocm-smi")
            .arg("--showproductname")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    #[cfg(target_os = "windows")]
    {
        // Use PowerShell to check for AMD/Radeon display adapters
        let ps_script =
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name";
        std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", ps_script])
            .output()
            .map(|o| {
                if !o.status.success() {
                    return false;
                }
                let stdout = String::from_utf8_lossy(&o.stdout).to_lowercase();
                stdout.contains("amd") || stdout.contains("radeon")
            })
            .unwrap_or(false)
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        false
    }
}

/// Check if the AMD GPU is an APU (integrated) rather than a dedicated GPU.
///
/// APUs don't support ROCm/HIP compute. They use shared system memory
/// instead of dedicated VRAM. We detect this by:
/// 1. Checking the GPU name for APU indicators (via athenas_core::is_apu_name)
/// 2. Checking if VRAM is 0 or very low (APUs report 0 or shared VRAM)
fn is_amd_apu() -> bool {
    // Get GPU names and VRAM info using platform-appropriate method
    let gpu_infos = get_amd_gpu_infos();

    for (name, vram_total_mb) in gpu_infos {
        // Method 1: check GPU name for APU indicators
        if athenas_core::is_apu_name(&name) {
            return true;
        }

        // Method 2: check VRAM — dedicated GPUs have at least 1 GB VRAM
        // APUs report 0 or very small amounts
        if vram_total_mb < 1024 {
            return true;
        }
    }

    false
}

/// Get AMD GPU information (name, VRAM in MB) using platform-appropriate tools.
/// Returns a list of (gpu_name, vram_total_mb) tuples.
fn get_amd_gpu_infos() -> Vec<(String, u64)> {
    #[cfg(target_os = "linux")]
    {
        get_amd_gpu_infos_linux()
    }
    #[cfg(target_os = "windows")]
    {
        get_amd_gpu_infos_windows()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Vec::new()
    }
}

#[cfg(target_os = "linux")]
fn get_amd_gpu_infos_linux() -> Vec<(String, u64)> {
    let mut result = Vec::new();

    // Get GPU names from rocm-smi
    if let Ok(output) = std::process::Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--json"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                if let Some(obj) = json.as_object() {
                    for (_key, value) in obj {
                        let name = value
                            .get("Card series")
                            .or_else(|| value.get("Card model"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let vram_total_mb = value
                            .get("VRAM Total Memory (B)")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok())
                            .map(|b| b / (1024 * 1024))
                            .unwrap_or(0);
                        result.push((name, vram_total_mb));
                    }
                }
            }
        }
    }

    result
}

#[cfg(target_os = "windows")]
fn get_amd_gpu_infos_windows() -> Vec<(String, u64)> {
    let mut result = Vec::new();

    // Use PowerShell to get GPU name and VRAM via CIM/WMI
    // This works on all modern Windows versions
    let ps_script =
        "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json";

    if let Ok(output) = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", ps_script])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                // Can be a single object or an array
                let gpus = if let Some(arr) = json.as_array() {
                    arr.clone()
                } else {
                    vec![json]
                };

                for gpu in gpus {
                    let name = gpu
                        .get("Name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // AdapterRAM is u32 and wraps for >4GB, but for APU
                    // detection it's usually 0 or small anyway
                    let vram_total_mb = gpu
                        .get("AdapterRAM")
                        .and_then(|v| v.as_u64())
                        .map(|b| b / (1024 * 1024))
                        .unwrap_or(0);
                    result.push((name, vram_total_mb));
                }
            }
        }
    }

    // Filter to only AMD/Radeon GPUs
    result
        .into_iter()
        .filter(|(name, _)| {
            let lower = name.to_lowercase();
            lower.contains("amd") || lower.contains("radeon")
        })
        .collect()
}

/// Check if ROCm is installed (rocm-smi works).
fn detect_rocm() -> bool {
    std::process::Command::new("rocm-smi")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Try to install Vulkan libraries on Linux (libvulkan1, mesa-vulkan-drivers).
/// This enables GPU acceleration on systems with AMD/NVIDIA GPUs but missing
/// Vulkan loader libraries.
pub async fn try_install_vulkan_libs_pub() -> bool {
    try_install_vulkan_libs_inner().await
}

async fn try_install_vulkan_libs_inner() -> bool {
    // Detect package manager and install Vulkan libraries
    let managers: [(&str, &[&str]); 5] = [
        (
            "apt-get",
            &[
                "apt-get",
                "install",
                "-y",
                "libvulkan1",
                "mesa-vulkan-drivers",
            ],
        ),
        (
            "dnf",
            &[
                "dnf",
                "install",
                "-y",
                "vulkan-loader",
                "mesa-vulkan-drivers",
            ],
        ),
        (
            "yum",
            &[
                "yum",
                "install",
                "-y",
                "vulkan-loader",
                "mesa-vulkan-drivers",
            ],
        ),
        (
            "pacman",
            &[
                "pacman",
                "-S",
                "--noconfirm",
                "vulkan-icd-loader",
                "vulkan-mesa-layers",
            ],
        ),
        (
            "apk",
            &["apk", "add", "vulkan-loader", "mesa-vulkan-drivers"],
        ),
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

            info!("Installing Vulkan libraries via {}...", name);
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
                        info!("Vulkan libraries installed successfully via {}", name);
                        // Verify libvulkan.so is now available
                        detect_vulkan_support()
                    } else {
                        warn!(
                            "Failed to install Vulkan libraries via {}: {}",
                            name,
                            String::from_utf8_lossy(&output.stderr)
                        );
                        false
                    }
                }
                Err(e) => {
                    warn!("Failed to run {}: {}", name, e);
                    false
                }
            };
        }
    }

    warn!("No supported package manager found to install Vulkan libraries");
    false
}

/// Check if Vulkan is available.
/// On Linux: checks for vulkaninfo, libvulkan.so, or NVIDIA driver.
/// On Windows: checks for vulkaninfo or vulkan-1.dll in System32.
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
        // Method 1: try vulkaninfo (comes with Vulkan SDK)
        if std::process::Command::new("vulkaninfo")
            .arg("--summary")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return true;
        }
        // Method 2: check for vulkan-1.dll in System32/SysWOW64
        // The Vulkan loader DLL is installed by GPU drivers (AMD, NVIDIA, Intel)
        let win_dir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
        let dll_paths = [
            format!("{}\\System32\\vulkan-1.dll", win_dir),
            format!("{}\\SysWOW64\\vulkan-1.dll", win_dir),
        ];
        for path in &dll_paths {
            if std::path::Path::new(path).exists() {
                return true;
            }
        }
        // Method 3: check for GPU-specific Vulkan ICD drivers
        let icd_paths = [
            format!("{}\\System32\\vulkan_radeon.dll", win_dir), // AMD
            format!("{}\\System32\\vulkan_intel.dll", win_dir),  // Intel
            format!("{}\\System32\\nvoglv64.dll", win_dir),      // NVIDIA
            format!("{}\\System32\\nvcuda.dll", win_dir),        // NVIDIA CUDA
        ];
        for path in &icd_paths {
            if std::path::Path::new(path).exists() {
                return true;
            }
        }
        false
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
    let desired_asset_suffix = platform_asset_name().await.ok_or_else(|| {
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

            // On Linux, if libgomp.so.1 is missing, try to auto-install it
            #[cfg(target_os = "linux")]
            {
                if stderr.contains("libgomp.so.1") {
                    info!("libgomp.so.1 missing during verification, attempting auto-install...");
                    if crate::llama_cpp::try_install_libgomp().await {
                        info!("libgomp1 installed, re-verifying llama-server...");
                        let reverify = verify_cmd.output().await;
                        if let Ok(rv) = reverify {
                            if rv.status.success() {
                                info!("llama-server verified successfully after libgomp install");
                                // Success — continue to variant marker
                            } else {
                                let rv_stderr = String::from_utf8_lossy(&rv.stderr);
                                return Err(AthenasError::Backend(format!(
                                    "llama-server still fails after libgomp install (exit code: {:?}).\n\
                                     stderr: {}\n\
                                     Try: ldd {}",
                                    rv.status.code(),
                                    rv_stderr,
                                    extracted_path.display()
                                )));
                            }
                        } else {
                            return Err(AthenasError::Backend(
                                "Cannot re-execute llama-server after libgomp install".into(),
                            ));
                        }
                    } else {
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
                } else if stderr.contains("libvulkan") {
                    info!("libvulkan missing during verification, attempting auto-install...");
                    if try_install_vulkan_libs_pub().await {
                        info!("Vulkan libraries installed, re-verifying llama-server...");
                        let reverify = verify_cmd.output().await;
                        if let Ok(rv) = reverify {
                            if rv.status.success() {
                                info!("llama-server verified successfully after Vulkan install");
                                // Success — continue to variant marker
                            } else {
                                let rv_stderr = String::from_utf8_lossy(&rv.stderr);
                                return Err(AthenasError::Backend(format!(
                                    "llama-server still fails after Vulkan install (exit code: {:?}).\n\
                                     stderr: {}\n\
                                     Try: ldd {}",
                                    rv.status.code(),
                                    rv_stderr,
                                    extracted_path.display()
                                )));
                            }
                        } else {
                            return Err(AthenasError::Backend(
                                "Cannot re-execute llama-server after Vulkan install".into(),
                            ));
                        }
                    } else {
                        return Err(AthenasError::Backend(format!(
                            "Downloaded llama-server failed to run (exit code: {:?}).\n\
                             stdout: {}\n\
                             stderr: {}\n\
                             Vulkan libraries are missing and could not be auto-installed.\n\
                             Try: ldd {}\n\
                             On Ubuntu/Debian: apt install -y libvulkan1 mesa-vulkan-drivers",
                            output.status.code(),
                            stdout,
                            stderr,
                            extracted_path.display()
                        )));
                    }
                } else {
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
            }

            #[cfg(not(target_os = "linux"))]
            {
                return Err(AthenasError::Backend(format!(
                    "Downloaded llama-server failed to run (exit code: {:?}).\n\
                     stdout: {}\n\
                     stderr: {}",
                    output.status.code(),
                    stdout,
                    stderr,
                )));
            }
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

// ===========================================================================
// whisper.cpp auto-download
// ===========================================================================

const WHISPER_CPP_REPO: &str = "ggml-org/whisper.cpp";
const WHISPER_CPP_VERSION: &str = "v1.7.6";

/// Detect the platform-appropriate asset name for whisper.cpp releases.
fn whisper_platform_asset_name() -> Option<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    match (os, arch) {
        ("linux", "x86_64") => Some("whisper-bin-ubuntu-x64.tar.gz".to_string()),
        ("linux", "aarch64") => Some("whisper-bin-ubuntu-arm64.tar.gz".to_string()),
        ("windows", "x86_64") => Some("whisper-bin-x64.zip".to_string()),
        ("windows", "x86") => Some("whisper-bin-Win32.zip".to_string()),
        ("macos", "x86_64") | ("macos", "aarch64") => {
            // whisper.cpp doesn't ship a macOS CLI binary in releases.
            // Users need to install via Homebrew: `brew install whisper-cpp`
            // or build from source.
            None
        }
        _ => None,
    }
}

/// Auto-download and install whisper-cli to ~/.athenas/bin/
pub async fn ensure_whisper_cli() -> Result<PathBuf> {
    let bin_dir = athenas_bin_dir()?;

    let cli_name = if std::env::consts::OS == "windows" {
        "whisper-cli.exe"
    } else {
        "whisper-cli"
    };

    let cli_path = bin_dir.join(cli_name);

    // Already installed?
    if cli_path.exists() {
        return Ok(cli_path);
    }

    let asset_suffix = whisper_platform_asset_name().ok_or_else(|| {
        AthenasError::Backend(format!(
            "No prebuilt whisper-cli available for {} {}. \
                 On macOS, install via: brew install whisper-cpp",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
    })?;

    info!("whisper-cli not found, auto-downloading...");

    // whisper.cpp release assets are named differently from llama.cpp.
    // They use: whisper-bin-ubuntu-x64.tar.gz (no tag prefix in the filename)
    let download_url = format!(
        "https://github.com/{}/releases/download/{}/{}",
        WHISPER_CPP_REPO, WHISPER_CPP_VERSION, asset_suffix
    );

    let data = download_file(&download_url).await?;
    info!(
        "Downloaded {} ({} MB), extracting...",
        asset_suffix,
        data.len() / (1024 * 1024)
    );

    let is_zip = asset_suffix.ends_with(".zip");
    let extracted_path = if is_zip {
        extract_zip_whisper(&data, &bin_dir)?
    } else {
        extract_tar_gz_whisper(&data, &bin_dir)?
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

    info!("whisper-cli installed to {}", extracted_path.display());
    Ok(extracted_path)
}

/// Extract whisper-cli from a tar.gz archive.
fn extract_tar_gz_whisper(data: &[u8], bin_dir: &std::path::Path) -> Result<PathBuf> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    let decoder = GzDecoder::new(data);
    let mut archive = Archive::new(decoder);

    let cli_name = if std::env::consts::OS == "windows" {
        "whisper-cli.exe"
    } else {
        "whisper-cli"
    };

    let mut cli_path = None;

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

        let dest = bin_dir.join(&file_name);

        #[cfg(unix)]
        {
            entry.set_preserve_permissions(true);
        }

        entry.unpack(&dest).map_err(|e| {
            AthenasError::Backend(format!("Failed to extract {}: {}", file_name, e))
        })?;

        if file_name == cli_name {
            cli_path = Some(dest);
        }
    }

    cli_path.ok_or_else(|| AthenasError::Backend("whisper-cli not found in archive".into()))
}

/// Extract whisper-cli from a zip archive.
fn extract_zip_whisper(data: &[u8], bin_dir: &std::path::Path) -> Result<PathBuf> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| AthenasError::Backend(format!("Failed to open zip: {}", e)))?;

    let cli_name = "whisper-cli.exe";
    let mut cli_path = None;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| AthenasError::Backend(format!("Failed to read zip entry: {}", e)))?;

        let name = file.name().to_string();
        if name.ends_with('/') {
            continue;
        }

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

        if file_name == cli_name {
            cli_path = Some(dest);
        }
    }

    cli_path.ok_or_else(|| AthenasError::Backend("whisper-cli.exe not found in zip".into()))
}
