use athenas_core::{
    AppConfig, BackendType, HardwareInfo, ModelInfo as RegistryModelInfo, ModelRegistry,
};
use athenas_inference::{Backend, BackendFactory, ModelLoadConfig};

#[derive(Debug, Clone, PartialEq)]
pub enum ServerPhase {
    Configuring,
    LoadingModel,
    Running,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigField {
    // Model selection
    ModelSelection,
    // Server config
    Host,
    Port,
    ApiKey,
    MaxConcurrent,
    RateLimit,
    TimeoutSecs,
    MaxBodySize,
    CorsEnabled,
    MetricsEnabled,
    CompressionEnabled,
    // Inference / optimization
    Backend,
    GpuLayers,
    ContextSize,
    BatchSize,
    Threads,
    FlashAttention,
    // Actions
    StartServer,
    StopServer,
}

impl ConfigField {
    pub fn all() -> Vec<ConfigField> {
        vec![
            ConfigField::ModelSelection,
            ConfigField::Host,
            ConfigField::Port,
            ConfigField::ApiKey,
            ConfigField::MaxConcurrent,
            ConfigField::RateLimit,
            ConfigField::TimeoutSecs,
            ConfigField::MaxBodySize,
            ConfigField::CorsEnabled,
            ConfigField::MetricsEnabled,
            ConfigField::CompressionEnabled,
            ConfigField::Backend,
            ConfigField::GpuLayers,
            ConfigField::ContextSize,
            ConfigField::BatchSize,
            ConfigField::Threads,
            ConfigField::FlashAttention,
            ConfigField::StartServer,
            ConfigField::StopServer,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            ConfigField::ModelSelection => "Model",
            ConfigField::Host => "Host",
            ConfigField::Port => "Port",
            ConfigField::ApiKey => "API Key",
            ConfigField::MaxConcurrent => "Max Concurrent",
            ConfigField::RateLimit => "Rate Limit (req/s)",
            ConfigField::TimeoutSecs => "Timeout (secs)",
            ConfigField::MaxBodySize => "Max Body Size (MB)",
            ConfigField::CorsEnabled => "CORS",
            ConfigField::MetricsEnabled => "Metrics",
            ConfigField::CompressionEnabled => "Compression",
            ConfigField::Backend => "Backend",
            ConfigField::GpuLayers => "GPU Layers",
            ConfigField::ContextSize => "Context Size",
            ConfigField::BatchSize => "Batch Size",
            ConfigField::Threads => "Threads",
            ConfigField::FlashAttention => "Flash Attention",
            ConfigField::StartServer => "Start Server",
            ConfigField::StopServer => "Stop Server",
        }
    }

    pub fn section(&self) -> &'static str {
        match self {
            ConfigField::ModelSelection => "MODEL",
            ConfigField::Host
            | ConfigField::Port
            | ConfigField::ApiKey
            | ConfigField::MaxConcurrent
            | ConfigField::RateLimit
            | ConfigField::TimeoutSecs
            | ConfigField::MaxBodySize
            | ConfigField::CorsEnabled
            | ConfigField::MetricsEnabled
            | ConfigField::CompressionEnabled => "SERVER",
            ConfigField::Backend
            | ConfigField::GpuLayers
            | ConfigField::ContextSize
            | ConfigField::BatchSize
            | ConfigField::Threads
            | ConfigField::FlashAttention => "OPTIMIZATION",
            ConfigField::StartServer | ConfigField::StopServer => "ACTION",
        }
    }

    pub fn is_editable(&self) -> bool {
        !matches!(
            self,
            ConfigField::ModelSelection | ConfigField::StartServer | ConfigField::StopServer
        )
    }

    pub fn is_toggle(&self) -> bool {
        matches!(
            self,
            ConfigField::CorsEnabled
                | ConfigField::MetricsEnabled
                | ConfigField::CompressionEnabled
                | ConfigField::FlashAttention
        )
    }

    pub fn is_action(&self) -> bool {
        matches!(self, ConfigField::StartServer | ConfigField::StopServer)
    }
}

pub struct ServerPanelState {
    pub fields: Vec<ConfigField>,
    pub selected: usize,
    pub editing: bool,
    pub edit_buffer: String,

    // Model selection
    pub models: Vec<RegistryModelInfo>,
    pub model_selected: usize,

    // Config values (edited copies)
    pub host: String,
    pub port: u16,
    pub api_key: String,
    pub max_concurrent: u32,
    pub rate_limit: u32,
    pub timeout_secs: u64,
    pub max_body_size: u32,
    pub cors_enabled: bool,
    pub metrics_enabled: bool,
    pub compression_enabled: bool,

    // Optimization
    pub backend: BackendType,
    pub gpu_layers: i32,
    pub context_size: u32,
    pub batch_size: u32,
    pub threads: u32,
    pub flash_attention: bool,

    // Runtime state
    pub phase: ServerPhase,
    pub status_message: Option<String>,
    pub server_url: Option<String>,
    pub loaded_model_name: Option<String>,
    pub loaded_backend_name: Option<String>,

    // Hardware info for display
    pub hardware: HardwareInfo,
}

impl ServerPanelState {
    pub fn new(config: &AppConfig, hardware: HardwareInfo) -> Self {
        let registry = ModelRegistry::new(config.paths.models_dir.clone());
        let models = registry.list_local_models().unwrap_or_default();

        Self {
            fields: ConfigField::all(),
            selected: 0,
            editing: false,
            edit_buffer: String::new(),
            models,
            model_selected: 0,
            host: config.server.default_host.clone(),
            port: config.server.default_port,
            api_key: config.server.api_key.clone().unwrap_or_default(),
            max_concurrent: config.server.max_concurrent_requests,
            rate_limit: config.server.rate_limit_per_second,
            timeout_secs: config.server.request_timeout_secs,
            max_body_size: config.server.max_body_size_mb,
            cors_enabled: config.server.cors_enabled,
            metrics_enabled: config.server.enable_metrics,
            compression_enabled: config.server.enable_compression,
            backend: config.inference.default_backend,
            gpu_layers: config.inference.default_gpu_layers,
            context_size: config.inference.default_context_size,
            batch_size: config.inference.default_batch_size,
            threads: config.inference.default_threads,
            flash_attention: config.inference.flash_attention,
            phase: ServerPhase::Configuring,
            status_message: None,
            server_url: None,
            loaded_model_name: None,
            loaded_backend_name: None,
            hardware,
        }
    }

    pub fn refresh_models(&mut self, config: &AppConfig) {
        let registry = ModelRegistry::new(config.paths.models_dir.clone());
        self.models = registry.list_local_models().unwrap_or_default();
    }

    pub fn next(&mut self) {
        if !self.editing {
            self.selected = (self.selected + 1) % self.fields.len();
        }
    }

    pub fn previous(&mut self) {
        if !self.editing {
            if self.selected == 0 {
                self.selected = self.fields.len() - 1;
            } else {
                self.selected -= 1;
            }
        }
    }

    pub fn current_field(&self) -> &ConfigField {
        &self.fields[self.selected]
    }

    pub fn field_value(&self, field: &ConfigField) -> String {
        match field {
            ConfigField::ModelSelection => {
                if self.models.is_empty() {
                    "No models found".to_string()
                } else {
                    self.models
                        .get(self.model_selected)
                        .map(|m| {
                            let q = m
                                .quantization
                                .as_ref()
                                .map(|q| format!(" [{}]", q))
                                .unwrap_or_default();
                            format!("{}{} ({})", m.name, q, m.format_size())
                        })
                        .unwrap_or_default()
                }
            }
            ConfigField::Host => self.host.clone(),
            ConfigField::Port => self.port.to_string(),
            ConfigField::ApiKey => {
                if self.api_key.is_empty() {
                    "(none)".to_string()
                } else {
                    "[hidden]".to_string()
                }
            }
            ConfigField::MaxConcurrent => self.max_concurrent.to_string(),
            ConfigField::RateLimit => self.rate_limit.to_string(),
            ConfigField::TimeoutSecs => self.timeout_secs.to_string(),
            ConfigField::MaxBodySize => self.max_body_size.to_string(),
            ConfigField::CorsEnabled => on_off(self.cors_enabled),
            ConfigField::MetricsEnabled => on_off(self.metrics_enabled),
            ConfigField::CompressionEnabled => on_off(self.compression_enabled),
            ConfigField::Backend => self.backend.to_string(),
            ConfigField::GpuLayers => {
                if self.gpu_layers < 0 {
                    "all".to_string()
                } else {
                    self.gpu_layers.to_string()
                }
            }
            ConfigField::ContextSize => self.context_size.to_string(),
            ConfigField::BatchSize => self.batch_size.to_string(),
            ConfigField::Threads => self.threads.to_string(),
            ConfigField::FlashAttention => on_off(self.flash_attention),
            ConfigField::StartServer => {
                if self.phase == ServerPhase::Running {
                    "Server is running".to_string()
                } else {
                    "Press Enter to start".to_string()
                }
            }
            ConfigField::StopServer => "Press Enter to stop".to_string(),
        }
    }

    pub fn field_hint(&self, field: &ConfigField) -> &'static str {
        match field {
            ConfigField::ModelSelection => "Up/Down to select from local models",
            ConfigField::Host => "0.0.0.0 for all interfaces, 127.0.0.1 for local",
            ConfigField::Port => "Port number (e.g. 8080)",
            ConfigField::ApiKey => "Leave empty for no auth, or set a secret key",
            ConfigField::MaxConcurrent => "Max simultaneous inference requests",
            ConfigField::RateLimit => "Token bucket: requests per second per IP",
            ConfigField::TimeoutSecs => "Kill stuck requests after N seconds",
            ConfigField::MaxBodySize => "Reject request bodies larger than N MB",
            ConfigField::CorsEnabled => "Allow cross-origin requests",
            ConfigField::MetricsEnabled => "Expose /metrics endpoint (Prometheus)",
            ConfigField::CompressionEnabled => "gzip response compression",
            ConfigField::Backend => "auto, llama.cpp, or vllm",
            ConfigField::GpuLayers => "-1 = all layers on GPU, 0 = CPU only",
            ConfigField::ContextSize => "Context window size in tokens",
            ConfigField::BatchSize => "Prompt processing batch size",
            ConfigField::Threads => "CPU threads (0 = auto)",
            ConfigField::FlashAttention => "Enable flash attention if supported",
            ConfigField::StartServer => "Loads model and starts the API server",
            ConfigField::StopServer => "Stops the running server",
        }
    }

    pub fn start_edit(&mut self) {
        let field = self.current_field().clone();
        if !field.is_editable() {
            return;
        }
        self.edit_buffer = self.field_value(&field);
        if field == ConfigField::ApiKey && self.edit_buffer == "[hidden]" {
            self.edit_buffer.clear();
            self.edit_buffer.push_str("[type to replace]");
        }
        self.editing = true;
    }

    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
    }

    pub fn save_edit(&mut self) -> Result<(), String> {
        let field = self.current_field().clone();
        let value = self.edit_buffer.trim().to_string();

        // Don't save if it's the placeholder
        if value == "[type to replace]" {
            self.editing = false;
            self.edit_buffer.clear();
            return Ok(());
        }

        match field {
            ConfigField::Host => {
                if value.is_empty() {
                    return Err("Host cannot be empty".to_string());
                }
                self.host = value;
            }
            ConfigField::Port => {
                self.port = value.parse::<u16>().map_err(|_| "Invalid port number")?;
            }
            ConfigField::ApiKey => {
                self.api_key = value;
            }
            ConfigField::MaxConcurrent => {
                self.max_concurrent = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::RateLimit => {
                self.rate_limit = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::TimeoutSecs => {
                self.timeout_secs = value
                    .parse::<u64>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::MaxBodySize => {
                self.max_body_size = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::Backend => {
                self.backend = match value.to_lowercase().as_str() {
                    "auto" => BackendType::Auto,
                    "llama.cpp" | "llamacpp" | "llama" => BackendType::LlamaCpp,
                    "vllm" => BackendType::Vllm,
                    _ => return Err("Must be: auto, llama.cpp, or vllm".to_string()),
                };
            }
            ConfigField::GpuLayers => {
                self.gpu_layers = if value == "all" || value == "-1" {
                    -1
                } else {
                    value
                        .parse::<i32>()
                        .map_err(|_| "Must be a number or 'all'")?
                };
            }
            ConfigField::ContextSize => {
                self.context_size = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::BatchSize => {
                self.batch_size = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::Threads => {
                self.threads = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
            }
            _ => {}
        }

        self.editing = false;
        self.edit_buffer.clear();
        Ok(())
    }

    pub fn toggle(&mut self) {
        let field = self.current_field().clone();
        match field {
            ConfigField::CorsEnabled => self.cors_enabled = !self.cors_enabled,
            ConfigField::MetricsEnabled => self.metrics_enabled = !self.metrics_enabled,
            ConfigField::CompressionEnabled => {
                self.compression_enabled = !self.compression_enabled;
            }
            ConfigField::FlashAttention => self.flash_attention = !self.flash_attention,
            _ => {}
        }
    }

    pub fn select_model_next(&mut self) {
        if !self.models.is_empty() {
            self.model_selected = (self.model_selected + 1) % self.models.len();
        }
    }

    pub fn select_model_prev(&mut self) {
        if !self.models.is_empty() {
            if self.model_selected == 0 {
                self.model_selected = self.models.len() - 1;
            } else {
                self.model_selected -= 1;
            }
        }
    }

    pub fn selected_model_path(&self) -> Option<String> {
        self.models
            .get(self.model_selected)
            .map(|m| m.file_path.to_string_lossy().to_string())
    }

    pub fn build_load_config(&self, model_path: &str) -> ModelLoadConfig {
        ModelLoadConfig {
            model_path: model_path.to_string(),
            gpu_layers: self.gpu_layers,
            context_size: self.context_size,
            batch_size: self.batch_size,
            threads: self.threads,
            flash_attention: self.flash_attention,
            use_mmap: true,
            use_mlock: false,
        }
    }

    pub fn build_app_config(&self, base: &AppConfig) -> AppConfig {
        let mut config = base.clone();
        config.server.default_host = self.host.clone();
        config.server.default_port = self.port;
        config.server.api_key = if self.api_key.is_empty() {
            None
        } else {
            Some(self.api_key.clone())
        };
        config.server.max_concurrent_requests = self.max_concurrent;
        config.server.rate_limit_per_second = self.rate_limit;
        config.server.request_timeout_secs = self.timeout_secs;
        config.server.max_body_size_mb = self.max_body_size;
        config.server.cors_enabled = self.cors_enabled;
        config.server.enable_metrics = self.metrics_enabled;
        config.server.enable_compression = self.compression_enabled;
        config.inference.default_backend = self.backend;
        config.inference.default_gpu_layers = self.gpu_layers;
        config.inference.default_context_size = self.context_size;
        config.inference.default_batch_size = self.batch_size;
        config.inference.default_threads = self.threads;
        config.inference.flash_attention = self.flash_attention;
        config
    }

    pub fn create_backend(&self) -> Result<Box<dyn Backend>, String> {
        BackendFactory::create(self.backend, &self.hardware)
            .map_err(|e| format!("Failed to create backend: {}", e))
    }
}

fn on_off(b: bool) -> String {
    if b {
        "ON".to_string()
    } else {
        "OFF".to_string()
    }
}
