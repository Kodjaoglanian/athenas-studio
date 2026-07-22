use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::errors::{AthenasError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub version: String,
    pub paths: PathsConfig,
    pub inference: InferenceConfig,
    pub server: ServerConfig,
    pub huggingface: HuggingFaceConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub models_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub data_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub default_backend: BackendType,
    pub default_gpu_layers: i32,
    pub default_context_size: u32,
    pub default_batch_size: u32,
    pub default_threads: u32,
    pub flash_attention: bool,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub default_max_tokens: u32,
    #[serde(default = "default_true")]
    pub streaming_enabled: bool,
    /// Enable reasoning/thinking mode for models that support it (Qwen3.5, DeepSeek R1, etc.)
    #[serde(default = "default_true")]
    pub reasoning_enabled: bool,
    /// Token budget for thinking: -1 for unrestricted, 0 for disabled, N>0 for specific budget
    #[serde(default = "default_reasoning_budget")]
    pub reasoning_budget: i32,
    /// MB of RAM to reserve for the OS/other apps. Model loading will not use more than (total - reserve).
    #[serde(default = "default_ram_reserve")]
    pub ram_reserve_mb: u64,
    /// Number of CPU cores to leave free for the system (0 = use all but 1)
    #[serde(default = "default_cpu_reserve")]
    pub cpu_reserve_cores: u32,
    /// When true, automatically cap threads/context/batch based on available hardware.
    /// When false, use the configured values as-is without auto-capping.
    #[serde(default = "default_true")]
    pub auto_resource_limits: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub default_host: String,
    pub default_port: u16,
    pub cors_enabled: bool,
    pub api_key: Option<String>,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: u32,
    #[serde(default = "default_rate_limit")]
    pub rate_limit_per_second: u32,
    #[serde(default = "default_timeout_secs")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_max_body_size")]
    pub max_body_size_mb: u32,
    #[serde(default = "default_true")]
    pub enable_metrics: bool,
    #[serde(default = "default_true")]
    pub enable_compression: bool,
}

fn default_max_concurrent() -> u32 {
    10
}
fn default_rate_limit() -> u32 {
    20
}
fn default_timeout_secs() -> u64 {
    120
}
fn default_max_body_size() -> u32 {
    10
}
fn default_true() -> bool {
    true
}
fn default_reasoning_budget() -> i32 {
    -1
}
fn default_ram_reserve() -> u64 {
    2048
}
fn default_cpu_reserve() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    pub token: Option<String>,
    pub default_revision: String,
    pub mirror_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file_logging: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    LlamaCpp,
    Vllm,
    Auto,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::LlamaCpp => write!(f, "llama.cpp"),
            BackendType::Vllm => write!(f, "vllm"),
            BackendType::Auto => write!(f, "auto"),
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let base = home.join(".athenas");

        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            paths: PathsConfig {
                models_dir: base.join("models"),
                cache_dir: base.join("cache"),
                data_dir: base.join("data"),
            },
            inference: InferenceConfig {
                default_backend: BackendType::Auto,
                default_gpu_layers: -1,
                default_context_size: 2048,
                default_batch_size: 256,
                default_threads: num_threads(),
                flash_attention: true,
                default_temperature: 0.7,
                default_top_p: 0.9,
                default_max_tokens: 2048,
                streaming_enabled: true,
                reasoning_enabled: true,
                reasoning_budget: -1,
                ram_reserve_mb: 2048,
                cpu_reserve_cores: 1,
                auto_resource_limits: true,
            },
            server: ServerConfig {
                default_host: "127.0.0.1".to_string(),
                default_port: 8080,
                cors_enabled: true,
                api_key: None,
                max_concurrent_requests: 10,
                rate_limit_per_second: 20,
                request_timeout_secs: 120,
                max_body_size_mb: 10,
                enable_metrics: true,
                enable_compression: true,
            },
            huggingface: HuggingFaceConfig {
                token: None,
                default_revision: "main".to_string(),
                mirror_url: None,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file_logging: false,
            },
        }
    }
}

fn num_threads() -> u32 {
    let total = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4);
    // Leave at least 1 core free for the system to prevent freezes
    total.saturating_sub(1).max(1)
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;
        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let config: AppConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            let config = AppConfig::default();
            config.save()?;
            Ok(config)
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content =
            toml::to_string_pretty(self).map_err(|e| AthenasError::Config(e.to_string()))?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    pub fn config_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| AthenasError::Config("Cannot determine home directory".to_string()))?;
        Ok(home.join(".athenas").join("config.toml"))
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(&self.paths.models_dir)?;
        std::fs::create_dir_all(&self.paths.cache_dir)?;
        std::fs::create_dir_all(&self.paths.data_dir)?;
        Ok(())
    }
}
