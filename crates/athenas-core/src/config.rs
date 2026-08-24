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
    /// GPU runtime to use: auto, cuda, rocm, vulkan, metal, cpu
    #[serde(default)]
    pub gpu_runtime: GpuRuntime,
    /// Which GPU device to use (0, 1, 2, ...). None = use default (usually 0).
    #[serde(default)]
    pub gpu_device: Option<u32>,
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
    /// LoRA adapter paths (comma-separated in config, stored as Vec)
    #[serde(default)]
    pub lora_paths: Vec<String>,
    /// Number of parallel decoding slots for batched inference
    #[serde(default = "default_parallel_slots_cfg")]
    pub parallel_slots: u32,
    /// Path to a draft model for speculative decoding (relative to models_dir
    /// or absolute). None = disabled.
    #[serde(default)]
    pub draft_model: Option<String>,
    /// Maximum number of tokens the draft model can propose per step.
    #[serde(default = "default_draft_max_tokens_cfg")]
    pub draft_max_tokens: u32,
    /// Minimum context size for the draft model.
    #[serde(default = "default_draft_min_ctx_cfg")]
    pub draft_min_ctx: u32,
}

fn default_draft_max_tokens_cfg() -> u32 {
    16
}

fn default_draft_min_ctx_cfg() -> u32 {
    512
}

fn default_parallel_slots_cfg() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub default_host: String,
    pub default_port: u16,
    pub cors_enabled: bool,
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
    /// Vector store configuration
    #[serde(default)]
    pub vector_store: VectorStoreServerConfig,
    /// IP filter: allowlist of IPs/CIDRs that can access the server. Empty = allow all.
    #[serde(default)]
    pub ip_allowlist: Vec<String>,
    /// IP filter: denylist of IPs/CIDRs that are blocked.
    #[serde(default)]
    pub ip_denylist: Vec<String>,
    /// OpenTelemetry tracing configuration
    #[serde(default)]
    pub otel: OtelConfig,
    /// When true, responses include `X-Queue-Position` and `X-Active-Requests`
    /// headers indicating the request's position in the queue when waiting
    /// for a semaphore permit. Does not change behavior — purely informational.
    #[serde(default)]
    pub queue_visibility: bool,
    /// Semantic cache configuration
    #[serde(default)]
    pub semantic_cache: SemanticCacheConfig,
}

fn default_max_concurrent() -> u32 {
    10
}
fn default_rate_limit() -> u32 {
    20
}
fn default_timeout_secs() -> u64 {
    300
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

/// Configuration for the integrated vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreServerConfig {
    /// Whether the vector store is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum number of documents to store (0 = unlimited).
    #[serde(default)]
    pub max_documents: usize,
    /// Number of results to return by default in search.
    #[serde(default = "default_vs_top_k")]
    pub default_top_k: usize,
}

fn default_vs_top_k() -> usize {
    5
}

/// Configuration for the semantic cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    /// Whether the semantic cache is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Minimum cosine similarity to consider a cache hit (0.0 - 1.0).
    #[serde(default = "default_cache_threshold")]
    pub similarity_threshold: f32,
    /// Time-to-live for cache entries in seconds.
    #[serde(default = "default_cache_ttl")]
    pub ttl_secs: u64,
    /// Maximum number of entries (LRU eviction when exceeded).
    #[serde(default = "default_cache_max_entries")]
    pub max_entries: usize,
}

fn default_cache_threshold() -> f32 {
    0.92
}

fn default_cache_ttl() -> u64 {
    3600
}

fn default_cache_max_entries() -> usize {
    1000
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            similarity_threshold: default_cache_threshold(),
            ttl_secs: default_cache_ttl(),
            max_entries: default_cache_max_entries(),
        }
    }
}

impl Default for VectorStoreServerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_documents: 0,
            default_top_k: 5,
        }
    }
}

/// Configuration for OpenTelemetry distributed tracing, logs, and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelConfig {
    /// Whether OpenTelemetry tracing is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// OTLP endpoint URL (e.g., "http://localhost:4317").
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Service name for traces.
    #[serde(default = "default_otel_service_name")]
    pub service_name: String,
    /// Sampling ratio (0.0 to 1.0).
    #[serde(default = "default_otel_sample_ratio")]
    pub sample_ratio: f64,
    /// Whether to export logs via OTLP (requires `enabled` + `endpoint`).
    #[serde(default = "default_true")]
    pub export_logs: bool,
    /// Whether to export metrics via OTLP (requires `enabled` + `endpoint`).
    #[serde(default = "default_true")]
    pub export_metrics: bool,
    /// Service namespace (e.g. "production", "staging").
    #[serde(default)]
    pub service_namespace: Option<String>,
    /// Deployment environment name (e.g. "production", "staging", "dev").
    #[serde(default)]
    pub environment: Option<String>,
    /// Service instance ID — auto-generated from hostname+PID if not set.
    #[serde(default)]
    pub service_instance_id: Option<String>,
}

fn default_otel_service_name() -> String {
    "athenas-studio".to_string()
}

fn default_otel_sample_ratio() -> f64 {
    1.0
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            service_name: default_otel_service_name(),
            sample_ratio: default_otel_sample_ratio(),
            export_logs: true,
            export_metrics: true,
            service_namespace: None,
            environment: None,
            service_instance_id: None,
        }
    }
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

/// GPU runtime to use for inference. Controls which backend library
/// llama.cpp/vLLM uses for GPU acceleration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum GpuRuntime {
    /// Auto-detect based on available hardware (CUDA > ROCm > Vulkan > Metal > CPU)
    #[default]
    Auto,
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm
    Rocm,
    /// Vulkan (cross-vendor)
    Vulkan,
    /// Apple Metal
    Metal,
    /// CPU only (no GPU acceleration)
    Cpu,
}

impl std::fmt::Display for GpuRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuRuntime::Auto => write!(f, "auto"),
            GpuRuntime::Cuda => write!(f, "cuda"),
            GpuRuntime::Rocm => write!(f, "rocm"),
            GpuRuntime::Vulkan => write!(f, "vulkan"),
            GpuRuntime::Metal => write!(f, "metal"),
            GpuRuntime::Cpu => write!(f, "cpu"),
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
                gpu_runtime: GpuRuntime::Auto,
                gpu_device: None,
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
                lora_paths: Vec::new(),
                parallel_slots: 4,
                draft_model: None,
                draft_max_tokens: 16,
                draft_min_ctx: 512,
            },
            server: ServerConfig {
                default_host: "127.0.0.1".to_string(),
                default_port: 8080,
                cors_enabled: true,
                max_concurrent_requests: 10,
                rate_limit_per_second: 20,
                request_timeout_secs: 300,
                max_body_size_mb: 10,
                enable_metrics: true,
                enable_compression: true,
                vector_store: VectorStoreServerConfig::default(),
                ip_allowlist: Vec::new(),
                ip_denylist: Vec::new(),
                otel: OtelConfig::default(),
                queue_visibility: false,
                semantic_cache: SemanticCacheConfig::default(),
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
