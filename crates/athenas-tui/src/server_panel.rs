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

#[derive(Debug, Clone)]
pub struct LoadedModelInfo {
    pub id: String,
    pub name: String,
    pub backend: String,
    pub is_default: bool,
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
    // Generation params
    MaxTokens,
    Temperature,
    TopP,
    Reasoning,
    ReasoningBudget,
    // Hardware protection
    RamReserve,
    CpuReserve,
    AutoResourceLimits,
    // Actions
    StartServer,
    StopServer,
    LoadAdditionalModel,
    UnloadModel,
    SetDefaultModel,
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
            ConfigField::MaxTokens,
            ConfigField::Temperature,
            ConfigField::TopP,
            ConfigField::Reasoning,
            ConfigField::ReasoningBudget,
            ConfigField::RamReserve,
            ConfigField::CpuReserve,
            ConfigField::AutoResourceLimits,
            ConfigField::StartServer,
            ConfigField::StopServer,
            ConfigField::LoadAdditionalModel,
            ConfigField::UnloadModel,
            ConfigField::SetDefaultModel,
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
            ConfigField::MaxTokens => "Max Tokens",
            ConfigField::Temperature => "Temperature",
            ConfigField::TopP => "Top P",
            ConfigField::Reasoning => "Reasoning/Thinking",
            ConfigField::ReasoningBudget => "Reasoning Budget",
            ConfigField::RamReserve => "RAM Reserve (MB)",
            ConfigField::CpuReserve => "CPU Reserve (cores)",
            ConfigField::AutoResourceLimits => "Auto Resource Limits",
            ConfigField::StartServer => "Start Server",
            ConfigField::StopServer => "Stop Server",
            ConfigField::LoadAdditionalModel => "Load Additional Model",
            ConfigField::UnloadModel => "Unload Model",
            ConfigField::SetDefaultModel => "Set Default Model",
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
            | ConfigField::FlashAttention
            | ConfigField::MaxTokens
            | ConfigField::Temperature
            | ConfigField::TopP
            | ConfigField::Reasoning
            | ConfigField::ReasoningBudget
            | ConfigField::RamReserve
            | ConfigField::CpuReserve
            | ConfigField::AutoResourceLimits => "OPTIMIZATION",
            ConfigField::StartServer
            | ConfigField::StopServer
            | ConfigField::LoadAdditionalModel
            | ConfigField::UnloadModel
            | ConfigField::SetDefaultModel => "ACTION",
        }
    }

    pub fn is_editable(&self) -> bool {
        !matches!(
            self,
            ConfigField::ModelSelection
                | ConfigField::StartServer
                | ConfigField::StopServer
                | ConfigField::LoadAdditionalModel
                | ConfigField::UnloadModel
                | ConfigField::SetDefaultModel
        )
    }

    pub fn is_toggle(&self) -> bool {
        matches!(
            self,
            ConfigField::CorsEnabled
                | ConfigField::MetricsEnabled
                | ConfigField::CompressionEnabled
                | ConfigField::FlashAttention
                | ConfigField::Reasoning
                | ConfigField::AutoResourceLimits
        )
    }

    pub fn is_action(&self) -> bool {
        matches!(
            self,
            ConfigField::StartServer
                | ConfigField::StopServer
                | ConfigField::LoadAdditionalModel
                | ConfigField::UnloadModel
                | ConfigField::SetDefaultModel
        )
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

    // Generation params
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub reasoning_enabled: bool,
    pub reasoning_budget: i32,

    // Hardware protection
    pub ram_reserve_mb: u64,
    pub cpu_reserve_cores: u32,
    pub auto_resource_limits: bool,

    // Runtime state
    pub phase: ServerPhase,
    pub status_message: Option<String>,
    pub server_url: Option<String>,
    pub loaded_model_name: Option<String>,
    pub loaded_backend_name: Option<String>,
    pub loaded_models: Vec<LoadedModelInfo>,
    pub unload_model_selected: usize,
    pub default_model_selected: usize,

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
            max_tokens: config.inference.default_max_tokens,
            temperature: config.inference.default_temperature,
            top_p: config.inference.default_top_p,
            reasoning_enabled: config.inference.reasoning_enabled,
            reasoning_budget: config.inference.reasoning_budget,
            ram_reserve_mb: config.inference.ram_reserve_mb,
            cpu_reserve_cores: config.inference.cpu_reserve_cores,
            auto_resource_limits: config.inference.auto_resource_limits,
            phase: ServerPhase::Configuring,
            status_message: None,
            server_url: None,
            loaded_model_name: None,
            loaded_backend_name: None,
            loaded_models: Vec::new(),
            unload_model_selected: 0,
            default_model_selected: 0,
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
            ConfigField::MaxTokens => self.max_tokens.to_string(),
            ConfigField::Temperature => self.temperature.to_string(),
            ConfigField::TopP => self.top_p.to_string(),
            ConfigField::Reasoning => on_off(self.reasoning_enabled),
            ConfigField::ReasoningBudget => self.reasoning_budget.to_string(),
            ConfigField::RamReserve => self.ram_reserve_mb.to_string(),
            ConfigField::CpuReserve => self.cpu_reserve_cores.to_string(),
            ConfigField::AutoResourceLimits => on_off(self.auto_resource_limits),
            ConfigField::StartServer => {
                if self.phase == ServerPhase::Running {
                    "Server is running".to_string()
                } else {
                    "Press Enter to start".to_string()
                }
            }
            ConfigField::StopServer => "Press Enter to stop".to_string(),
            ConfigField::LoadAdditionalModel => {
                if self.phase == ServerPhase::Running {
                    "Press Enter to load another model".to_string()
                } else {
                    "Start server first".to_string()
                }
            }
            ConfigField::UnloadModel => {
                if self.loaded_models.is_empty() {
                    "No models loaded".to_string()
                } else {
                    let m = &self.loaded_models
                        [self.unload_model_selected.min(self.loaded_models.len() - 1)];
                    format!(
                        "{}{} (Left/Right to select)",
                        m.name,
                        if m.is_default { " [default]" } else { "" }
                    )
                }
            }
            ConfigField::SetDefaultModel => {
                if self.loaded_models.is_empty() {
                    "No models loaded".to_string()
                } else {
                    let m = &self.loaded_models[self
                        .default_model_selected
                        .min(self.loaded_models.len() - 1)];
                    format!("{} (Left/Right to select)", m.name)
                }
            }
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
            ConfigField::MaxTokens => "Max tokens to generate (e.g. 2048)",
            ConfigField::Temperature => "0.0 - 2.0 (creativity)",
            ConfigField::TopP => "0.0 - 1.0 (nucleus sampling)",
            ConfigField::Reasoning => "Enable thinking mode (Qwen3.5, DeepSeek R1)",
            ConfigField::ReasoningBudget => "-1 = unlimited, 0 = off, N = token limit",
            ConfigField::RamReserve => "MB reserved for OS (e.g. 2048)",
            ConfigField::CpuReserve => "Cores to leave free (e.g. 1)",
            ConfigField::AutoResourceLimits => "Auto-cap threads/ctx/batch based on hardware",
            ConfigField::StartServer => "Loads model and starts the API server",
            ConfigField::StopServer => "Stops the running server",
            ConfigField::LoadAdditionalModel => "Load another model while server is running",
            ConfigField::UnloadModel => "Unload a model from memory (Left/Right to pick)",
            ConfigField::SetDefaultModel => "Set which model handles requests without model field",
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
            ConfigField::MaxTokens => {
                self.max_tokens = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::Temperature => {
                self.temperature = value
                    .parse::<f32>()
                    .map_err(|_| "Must be a float (0.0-2.0)")?;
            }
            ConfigField::TopP => {
                self.top_p = value
                    .parse::<f32>()
                    .map_err(|_| "Must be a float (0.0-1.0)")?;
            }
            ConfigField::ReasoningBudget => {
                self.reasoning_budget = value
                    .parse::<i32>()
                    .map_err(|_| "-1 = unlimited, 0 = off, N = token limit")?;
            }
            ConfigField::RamReserve => {
                self.ram_reserve_mb = value.parse::<u64>().map_err(|_| "Must be a number (MB)")?;
            }
            ConfigField::CpuReserve => {
                self.cpu_reserve_cores = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a number (cores)")?;
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
            ConfigField::Reasoning => self.reasoning_enabled = !self.reasoning_enabled,
            ConfigField::AutoResourceLimits => {
                self.auto_resource_limits = !self.auto_resource_limits;
            }
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
            reasoning_enabled: self.reasoning_enabled,
            reasoning_budget: self.reasoning_budget,
            mmproj_path: None,
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
        config.inference.default_max_tokens = self.max_tokens;
        config.inference.default_temperature = self.temperature;
        config.inference.default_top_p = self.top_p;
        config.inference.reasoning_enabled = self.reasoning_enabled;
        config.inference.reasoning_budget = self.reasoning_budget;
        config.inference.ram_reserve_mb = self.ram_reserve_mb;
        config.inference.cpu_reserve_cores = self.cpu_reserve_cores;
        config.inference.auto_resource_limits = self.auto_resource_limits;
        config
    }

    pub fn create_backend(&self) -> Result<Box<dyn Backend>, String> {
        BackendFactory::create(self.backend, &self.hardware)
            .map_err(|e| format!("Failed to create backend: {}", e))
    }

    pub fn unload_select_next(&mut self) {
        if !self.loaded_models.is_empty() {
            self.unload_model_selected =
                (self.unload_model_selected + 1) % self.loaded_models.len();
        }
    }

    pub fn unload_select_prev(&mut self) {
        if !self.loaded_models.is_empty() {
            if self.unload_model_selected == 0 {
                self.unload_model_selected = self.loaded_models.len() - 1;
            } else {
                self.unload_model_selected -= 1;
            }
        }
    }

    pub fn default_select_next(&mut self) {
        if !self.loaded_models.is_empty() {
            self.default_model_selected =
                (self.default_model_selected + 1) % self.loaded_models.len();
        }
    }

    pub fn default_select_prev(&mut self) {
        if !self.loaded_models.is_empty() {
            if self.default_model_selected == 0 {
                self.default_model_selected = self.loaded_models.len() - 1;
            } else {
                self.default_model_selected -= 1;
            }
        }
    }

    pub fn selected_unload_model_id(&self) -> Option<String> {
        self.loaded_models
            .get(
                self.unload_model_selected
                    .min(self.loaded_models.len().saturating_sub(1)),
            )
            .map(|m| m.id.clone())
    }

    pub fn selected_default_model_id(&self) -> Option<String> {
        self.loaded_models
            .get(
                self.default_model_selected
                    .min(self.loaded_models.len().saturating_sub(1)),
            )
            .map(|m| m.id.clone())
    }
}

fn on_off(b: bool) -> String {
    if b {
        "ON".to_string()
    } else {
        "OFF".to_string()
    }
}
