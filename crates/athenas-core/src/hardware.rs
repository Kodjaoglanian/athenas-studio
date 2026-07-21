use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::info;

use crate::errors::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpus: u32,
    pub memory_total_mb: u64,
    pub memory_available_mb: u64,
    pub gpus: Vec<GpuInfo>,
    pub has_cuda: bool,
    pub has_rocm: bool,
    pub has_vulkan: bool,
    pub has_metal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vendor: GpuVendor,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub driver_version: String,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuVendor::Nvidia => write!(f, "NVIDIA"),
            GpuVendor::Amd => write!(f, "AMD"),
            GpuVendor::Intel => write!(f, "Intel"),
            GpuVendor::Apple => write!(f, "Apple"),
            GpuVendor::Unknown => write!(f, "Unknown"),
        }
    }
}

impl std::fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} ({} VRAM: {}MB/{}MB)",
            self.index,
            self.name,
            self.vendor,
            self.vram_used_mb,
            self.vram_total_mb
        )
    }
}

impl std::fmt::Display for HardwareInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Hardware Info ===")?;
        writeln!(f, "CPUs: {}", self.cpus)?;
        writeln!(
            f,
            "Memory: {}MB / {}MB",
            self.memory_available_mb, self.memory_total_mb
        )?;
        writeln!(f, "CUDA: {}", if self.has_cuda { "yes" } else { "no" })?;
        writeln!(f, "ROCm: {}", if self.has_rocm { "yes" } else { "no" })?;
        writeln!(f, "Vulkan: {}", if self.has_vulkan { "yes" } else { "no" })?;
        writeln!(f, "Metal: {}", if self.has_metal { "yes" } else { "no" })?;
        if !self.gpus.is_empty() {
            writeln!(f, "GPUs:")?;
            for gpu in &self.gpus {
                writeln!(f, "  {}", gpu)?;
            }
        }
        Ok(())
    }
}

pub struct HardwareDetector;

impl HardwareDetector {
    pub fn detect() -> Result<HardwareInfo> {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4);

        let (memory_total_mb, memory_available_mb) = detect_memory();

        let mut gpus = Vec::new();
        let mut has_cuda = false;
        let mut has_rocm = false;
        let mut has_vulkan = false;
        let mut has_metal = false;

        // Detect NVIDIA GPUs via nvidia-smi
        if let Ok(nvidia_gpus) = detect_nvidia_gpus() {
            if !nvidia_gpus.is_empty() {
                has_cuda = true;
                info!("Found {} NVIDIA GPU(s) with CUDA support", nvidia_gpus.len());
            }
            gpus.extend(nvidia_gpus);
        }

        // Detect AMD GPUs via rocm-smi
        if let Ok(amd_gpus) = detect_amd_gpus() {
            if !amd_gpus.is_empty() {
                has_rocm = true;
                info!("Found {} AMD GPU(s) with ROCm support", amd_gpus.len());
            }
            gpus.extend(amd_gpus);
        }

        // Detect Vulkan support
        if detect_vulkan() {
            has_vulkan = true;
            info!("Vulkan support detected");
        }

        // Detect Metal (macOS only)
        if cfg!(target_os = "macos") {
            has_metal = true;
            info!("Metal support detected (macOS)");
        }

        // Sort GPUs by index
        gpus.sort_by_key(|g| g.index);

        Ok(HardwareInfo {
            cpus,
            memory_total_mb,
            memory_available_mb,
            gpus,
            has_cuda,
            has_rocm,
            has_vulkan,
            has_metal,
        })
    }

    pub fn recommend_gpu_layers(hw: &HardwareInfo, model_size_gb: f64) -> i32 {
        if hw.gpus.is_empty() {
            return 0;
        }

        let total_vram_mb: u64 = hw.gpus.iter().map(|g| g.vram_total_mb).sum();
        let total_vram_gb = total_vram_mb as f64 / 1024.0;
        let available_vram_gb = total_vram_gb * 0.85; // Leave 15% headroom

        if model_size_gb <= available_vram_gb {
            -1 // Full offload
        } else if model_size_gb <= total_vram_gb {
            // Partial offload — estimate layers
            let ratio = available_vram_gb / model_size_gb;
            (ratio * 100.0) as i32
        } else {
            // Model too large for VRAM, use partial
            let ratio = available_vram_gb / model_size_gb;
            (ratio * 80.0) as i32
        }
    }

    pub fn print_info(hw: &HardwareInfo) {
        println!("{}", hw);
    }
}

fn detect_memory() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total = 0u64;
            let mut available = 0u64;
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total = parse_meminfo_line(line);
                } else if line.starts_with("MemAvailable:") {
                    available = parse_meminfo_line(line);
                }
            }
            return (total / 1024, available / 1024);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                let total: u64 = s.trim().parse().unwrap_or(0);
                return (total / (1024 * 1024), total / (1024 * 1024));
            }
        }
    }

    (0, 0)
}

#[cfg(target_os = "linux")]
fn parse_meminfo_line(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn detect_nvidia_gpus() -> std::result::Result<Vec<GpuInfo>, ()> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|_| ())?;

    if !output.status.success() {
        return Err(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 6 {
            gpus.push(GpuInfo {
                index: parts[0].parse().unwrap_or(0),
                name: parts[1].to_string(),
                vendor: GpuVendor::Nvidia,
                vram_total_mb: parts[2].parse().unwrap_or(0),
                vram_used_mb: parts[3].parse().unwrap_or(0),
                driver_version: parts[4].to_string(),
                compute_capability: Some(parts[5].to_string()),
            });
        }
    }

    Ok(gpus)
}

fn detect_amd_gpus() -> std::result::Result<Vec<GpuInfo>, ()> {
    let output = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--json"])
        .output()
        .map_err(|_| ())?;

    if !output.status.success() {
        return Err(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
        if let Some(obj) = json.as_object() {
            for (key, value) in obj {
                let index: u32 = key.strip_prefix("card").and_then(|s| s.parse().ok()).unwrap_or(0);

                let name = value
                    .get("Card series")
                    .or_else(|| value.get("Card model"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("AMD GPU")
                    .to_string();

                let vram_total_mb = value
                    .get("VRAM Total Memory (B)")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(|b| b / (1024 * 1024))
                    .unwrap_or(0);

                let vram_used_mb = value
                    .get("VRAM Total Used Memory (B)")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(|b| b / (1024 * 1024))
                    .unwrap_or(0);

                gpus.push(GpuInfo {
                    index,
                    name,
                    vendor: GpuVendor::Amd,
                    vram_total_mb,
                    vram_used_mb,
                    driver_version: "ROCm".to_string(),
                    compute_capability: None,
                });
            }
        }
    }

    Ok(gpus)
}

fn detect_vulkan() -> bool {
    if cfg!(target_os = "linux") || cfg!(target_os = "windows") {
        Command::new("vulkaninfo")
            .arg("--summary")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    } else {
        false
    }
}
