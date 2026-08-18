use athenas_core::{
    estimate_model_ram_mb, AppConfig, BackendType, HardwareInfo, ModelInfo as RegistryModelInfo,
    ModelRegistry,
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

/// Info about a managed API key, fetched from the running server.
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub key_id: String,
    pub api_key: String,
    pub name: String,
    pub active: bool,
    pub rate_limit_per_minute: u32,
    pub daily_token_limit: u64,
    pub allowed_models: Vec<String>,
    pub created_at: String,
    // Usage metrics (today's stats)
    pub usage_requests: u64,
    pub usage_tokens_prompt: u64,
    pub usage_tokens_generated: u64,
    pub usage_tokens_total: u64,
    pub usage_date: String,
    pub rate_limit_remaining: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigField {
    // Model selection
    ModelSelection,
    // Server config
    Host,
    Port,
    MaxConcurrent,
    RateLimit,
    TimeoutSecs,
    MaxBodySize,
    CorsEnabled,
    MetricsEnabled,
    CompressionEnabled,
    QueueVisibility,
    // Inference / optimization
    Backend,
    GpuRuntime,
    GpuDevice,
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
    // Advanced inference
    ParallelSlots,
    LoraPaths,
    DraftModel,
    DraftMaxTokens,
    DraftMinCtx,
    // Vector store
    VectorStoreEnabled,
    VectorStoreMaxDocs,
    VectorStoreTopK,
    // OpenTelemetry
    OtelEnabled,
    OtelEndpoint,
    OtelServiceName,
    OtelSampleRatio,
    // IP filter
    IpAllowlist,
    IpDenylist,
    // Actions
    StartServer,
    StopServer,
    LoadAdditionalModel,
    UnloadModel,
    SetDefaultModel,
    // API Key Management (multi-tenant) — opens a modal
    ManageApiKeys,
}

impl ConfigField {
    pub fn all() -> Vec<ConfigField> {
        vec![
            ConfigField::ModelSelection,
            ConfigField::Host,
            ConfigField::Port,
            ConfigField::MaxConcurrent,
            ConfigField::RateLimit,
            ConfigField::TimeoutSecs,
            ConfigField::MaxBodySize,
            ConfigField::CorsEnabled,
            ConfigField::MetricsEnabled,
            ConfigField::CompressionEnabled,
            ConfigField::QueueVisibility,
            ConfigField::Backend,
            ConfigField::GpuRuntime,
            ConfigField::GpuDevice,
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
            ConfigField::ParallelSlots,
            ConfigField::LoraPaths,
            ConfigField::DraftModel,
            ConfigField::DraftMaxTokens,
            ConfigField::DraftMinCtx,
            ConfigField::VectorStoreEnabled,
            ConfigField::VectorStoreMaxDocs,
            ConfigField::VectorStoreTopK,
            ConfigField::OtelEnabled,
            ConfigField::OtelEndpoint,
            ConfigField::OtelServiceName,
            ConfigField::OtelSampleRatio,
            ConfigField::IpAllowlist,
            ConfigField::IpDenylist,
            ConfigField::StartServer,
            ConfigField::StopServer,
            ConfigField::LoadAdditionalModel,
            ConfigField::UnloadModel,
            ConfigField::SetDefaultModel,
            ConfigField::ManageApiKeys,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            ConfigField::ModelSelection => "Model",
            ConfigField::Host => "Host",
            ConfigField::Port => "Port",
            ConfigField::MaxConcurrent => "Max Concurrent",
            ConfigField::RateLimit => "Rate Limit (req/s)",
            ConfigField::TimeoutSecs => "Timeout (secs)",
            ConfigField::MaxBodySize => "Max Body Size (MB)",
            ConfigField::CorsEnabled => "CORS",
            ConfigField::MetricsEnabled => "Metrics",
            ConfigField::CompressionEnabled => "Compression",
            ConfigField::QueueVisibility => "Queue Info",
            ConfigField::Backend => "Backend",
            ConfigField::GpuRuntime => "GPU Runtime",
            ConfigField::GpuDevice => "GPU Device",
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
            ConfigField::ParallelSlots => "Parallel Slots",
            ConfigField::LoraPaths => "LoRA Adapters",
            ConfigField::DraftModel => "Draft Model",
            ConfigField::DraftMaxTokens => "Draft Max Tokens",
            ConfigField::DraftMinCtx => "Draft Min Ctx",
            ConfigField::VectorStoreEnabled => "Vector Store",
            ConfigField::VectorStoreMaxDocs => "VS Max Documents",
            ConfigField::VectorStoreTopK => "VS Default Top-K",
            ConfigField::OtelEnabled => "OpenTelemetry",
            ConfigField::OtelEndpoint => "OTLP Endpoint",
            ConfigField::OtelServiceName => "OTel Service Name",
            ConfigField::OtelSampleRatio => "OTel Sample Ratio",
            ConfigField::IpAllowlist => "IP Allowlist",
            ConfigField::IpDenylist => "IP Denylist",
            ConfigField::StartServer => "Start Server",
            ConfigField::StopServer => "Stop Server",
            ConfigField::LoadAdditionalModel => "Load Additional Model",
            ConfigField::UnloadModel => "Unload Model",
            ConfigField::SetDefaultModel => "Set Default Model",
            ConfigField::ManageApiKeys => "Manage API Keys",
        }
    }

    pub fn section(&self) -> &'static str {
        match self {
            ConfigField::ModelSelection => "MODEL",
            ConfigField::Host
            | ConfigField::Port
            | ConfigField::MaxConcurrent
            | ConfigField::RateLimit
            | ConfigField::TimeoutSecs
            | ConfigField::MaxBodySize
            | ConfigField::CorsEnabled
            | ConfigField::MetricsEnabled
            | ConfigField::CompressionEnabled
            | ConfigField::QueueVisibility => "SERVER",
            ConfigField::Backend
            | ConfigField::GpuRuntime
            | ConfigField::GpuDevice
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
            | ConfigField::AutoResourceLimits
            | ConfigField::ParallelSlots
            | ConfigField::LoraPaths
            | ConfigField::DraftModel
            | ConfigField::DraftMaxTokens
            | ConfigField::DraftMinCtx => "ADVANCED",
            ConfigField::VectorStoreEnabled
            | ConfigField::VectorStoreMaxDocs
            | ConfigField::VectorStoreTopK => "VECTOR STORE",
            ConfigField::OtelEnabled
            | ConfigField::OtelEndpoint
            | ConfigField::OtelServiceName
            | ConfigField::OtelSampleRatio => "TRACING",
            ConfigField::IpAllowlist | ConfigField::IpDenylist => "SECURITY",
            ConfigField::StartServer
            | ConfigField::StopServer
            | ConfigField::LoadAdditionalModel
            | ConfigField::UnloadModel
            | ConfigField::SetDefaultModel => "ACTION",
            ConfigField::ManageApiKeys => "API KEYS",
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
                | ConfigField::ManageApiKeys
        )
    }

    pub fn is_toggle(&self) -> bool {
        matches!(
            self,
            ConfigField::CorsEnabled
                | ConfigField::MetricsEnabled
                | ConfigField::CompressionEnabled
                | ConfigField::QueueVisibility
                | ConfigField::FlashAttention
                | ConfigField::Reasoning
                | ConfigField::AutoResourceLimits
                | ConfigField::VectorStoreEnabled
                | ConfigField::OtelEnabled
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
                | ConfigField::ManageApiKeys
        )
    }
}

/// Which field of the "Create API Key" form is being edited.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyEditField {
    Name,
    RateLimit,
    TokenLimit,
    AllowedModels,
}

/// Estimated memory footprint for loading the currently selected model
/// with the panel's current GPU/context settings. Shown in the hardware
/// banner so the user knows the cost *before* starting the server.
#[derive(Debug, Clone)]
pub struct LoadEstimate {
    /// Model file size in MB
    pub model_size_mb: u64,
    /// Estimated host RAM needed in MB
    pub ram_mb: u64,
    /// Estimated VRAM needed in MB (None = CPU-only load)
    pub vram_mb: Option<u64>,
    /// Available system RAM in MB (0 = detection failed)
    pub ram_available_mb: u64,
    /// Free VRAM in MB on the target GPU (when offloading)
    pub vram_free_mb: Option<u64>,
    /// True when the model is fully offloaded to the GPU (gpu_layers = -1)
    pub full_gpu_offload: bool,
    /// True when some layers are offloaded (0 < gpu_layers)
    pub partial_gpu_offload: bool,
    /// GPU layers setting the estimate is based on
    pub gpu_layers: i32,
    /// Whether the estimate fits in the available memory
    pub fits: bool,
    /// Whether it fits but with little headroom (>= 90% of available RAM)
    pub tight: bool,
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
    pub max_concurrent: u32,
    pub rate_limit: u32,
    pub timeout_secs: u64,
    pub max_body_size: u32,
    pub cors_enabled: bool,
    pub metrics_enabled: bool,
    pub compression_enabled: bool,
    pub queue_visibility: bool,

    // Optimization
    pub backend: BackendType,
    pub gpu_layers: i32,
    pub gpu_runtime: athenas_core::GpuRuntime,
    pub gpu_device: Option<u32>,
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

    // Advanced inference
    pub parallel_slots: u32,
    pub lora_paths: String,
    pub draft_model: String,
    pub draft_max_tokens: u32,
    pub draft_min_ctx: u32,

    // Vector store
    pub vs_enabled: bool,
    pub vs_max_documents: usize,
    pub vs_top_k: usize,

    // OpenTelemetry
    pub otel_enabled: bool,
    pub otel_endpoint: String,
    pub otel_service_name: String,
    pub otel_sample_ratio: f64,

    // IP filter
    pub ip_allowlist: String,
    pub ip_denylist: String,

    // Runtime state
    pub phase: ServerPhase,
    pub status_message: Option<String>,
    /// Whether the status message is an error (red) or info (cyan/green)
    pub status_is_error: bool,
    pub server_url: Option<String>,
    pub loaded_model_name: Option<String>,
    pub loaded_backend_name: Option<String>,
    pub loaded_models: Vec<LoadedModelInfo>,
    pub unload_model_selected: usize,
    pub default_model_selected: usize,

    // Hardware info for display
    pub hardware: HardwareInfo,

    // Multi-tenant API key management
    pub api_keys: Vec<ApiKeyInfo>,
    pub api_key_selected: usize,
    pub new_key_name: String,
    pub new_key_rate_limit: String,
    pub new_key_token_limit: String,
    pub new_key_allowed_models: String,
    pub editing_key_field: Option<KeyEditField>,
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
            max_concurrent: config.server.max_concurrent_requests,
            rate_limit: config.server.rate_limit_per_second,
            timeout_secs: config.server.request_timeout_secs,
            max_body_size: config.server.max_body_size_mb,
            cors_enabled: config.server.cors_enabled,
            metrics_enabled: config.server.enable_metrics,
            compression_enabled: config.server.enable_compression,
            queue_visibility: config.server.queue_visibility,
            backend: config.inference.default_backend,
            gpu_layers: config.inference.default_gpu_layers,
            gpu_runtime: config.inference.gpu_runtime,
            gpu_device: config.inference.gpu_device,
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
            parallel_slots: config.inference.parallel_slots,
            lora_paths: config.inference.lora_paths.join(", "),
            draft_model: config.inference.draft_model.clone().unwrap_or_default(),
            draft_max_tokens: config.inference.draft_max_tokens,
            draft_min_ctx: config.inference.draft_min_ctx,
            vs_enabled: config.server.vector_store.enabled,
            vs_max_documents: config.server.vector_store.max_documents,
            vs_top_k: config.server.vector_store.default_top_k,
            otel_enabled: config.server.otel.enabled,
            otel_endpoint: config.server.otel.endpoint.clone().unwrap_or_default(),
            otel_service_name: config.server.otel.service_name.clone(),
            otel_sample_ratio: config.server.otel.sample_ratio,
            ip_allowlist: config.server.ip_allowlist.join(", "),
            ip_denylist: config.server.ip_denylist.join(", "),
            phase: ServerPhase::Configuring,
            status_message: None,
            status_is_error: false,
            server_url: None,
            loaded_model_name: None,
            loaded_backend_name: None,
            loaded_models: Vec::new(),
            unload_model_selected: 0,
            default_model_selected: 0,
            hardware,
            api_keys: Vec::new(),
            api_key_selected: 0,
            new_key_name: String::new(),
            new_key_rate_limit: "60".to_string(),
            new_key_token_limit: "0".to_string(),
            new_key_allowed_models: String::new(),
            editing_key_field: None,
        }
    }

    pub fn refresh_models(&mut self, config: &AppConfig) {
        let registry = ModelRegistry::new(config.paths.models_dir.clone());
        self.models = registry.list_local_models().unwrap_or_default();
        // Keep the selection in bounds — models may have been deleted
        // or added since the last scan.
        if self.models.is_empty() {
            self.model_selected = 0;
        } else if self.model_selected >= self.models.len() {
            self.model_selected = self.models.len() - 1;
        }
    }

    /// Set an informational status message (cyan/green in the status bar).
    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
        self.status_is_error = false;
    }

    /// Set an error status message (red in the status bar).
    pub fn set_error(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
        self.status_is_error = true;
    }

    /// Clear the status message.
    pub fn clear_status(&mut self) {
        self.status_message = None;
        self.status_is_error = false;
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

    /// Jump forward `n` fields (PageDown) — stops at the last field.
    pub fn jump_forward(&mut self, n: usize) {
        if !self.editing {
            self.selected = (self.selected + n).min(self.fields.len() - 1);
        }
    }

    /// Jump backward `n` fields (PageUp) — stops at the first field.
    pub fn jump_back(&mut self, n: usize) {
        if !self.editing {
            self.selected = self.selected.saturating_sub(n);
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
            ConfigField::MaxConcurrent => self.max_concurrent.to_string(),
            ConfigField::RateLimit => self.rate_limit.to_string(),
            ConfigField::TimeoutSecs => self.timeout_secs.to_string(),
            ConfigField::MaxBodySize => self.max_body_size.to_string(),
            ConfigField::CorsEnabled => on_off(self.cors_enabled),
            ConfigField::MetricsEnabled => on_off(self.metrics_enabled),
            ConfigField::CompressionEnabled => on_off(self.compression_enabled),
            ConfigField::QueueVisibility => on_off(self.queue_visibility),
            ConfigField::Backend => self.backend.to_string(),
            ConfigField::GpuRuntime => self.gpu_runtime.to_string(),
            ConfigField::GpuDevice => match self.gpu_device {
                Some(d) => d.to_string(),
                None => "auto".to_string(),
            },
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
            ConfigField::ParallelSlots => self.parallel_slots.to_string(),
            ConfigField::LoraPaths => {
                if self.lora_paths.is_empty() {
                    "(none)".to_string()
                } else {
                    self.lora_paths.clone()
                }
            }
            ConfigField::DraftModel => {
                if self.draft_model.is_empty() {
                    "(none)".to_string()
                } else {
                    self.draft_model.clone()
                }
            }
            ConfigField::DraftMaxTokens => self.draft_max_tokens.to_string(),
            ConfigField::DraftMinCtx => self.draft_min_ctx.to_string(),
            ConfigField::VectorStoreEnabled => on_off(self.vs_enabled),
            ConfigField::VectorStoreMaxDocs => {
                if self.vs_max_documents == 0 {
                    "unlimited".to_string()
                } else {
                    self.vs_max_documents.to_string()
                }
            }
            ConfigField::VectorStoreTopK => self.vs_top_k.to_string(),
            ConfigField::OtelEnabled => on_off(self.otel_enabled),
            ConfigField::OtelEndpoint => {
                if self.otel_endpoint.is_empty() {
                    "(none)".to_string()
                } else {
                    self.otel_endpoint.clone()
                }
            }
            ConfigField::OtelServiceName => self.otel_service_name.clone(),
            ConfigField::OtelSampleRatio => self.otel_sample_ratio.to_string(),
            ConfigField::IpAllowlist => {
                if self.ip_allowlist.is_empty() {
                    "(allow all)".to_string()
                } else {
                    self.ip_allowlist.clone()
                }
            }
            ConfigField::IpDenylist => {
                if self.ip_denylist.is_empty() {
                    "(none)".to_string()
                } else {
                    self.ip_denylist.clone()
                }
            }
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
            ConfigField::ManageApiKeys => {
                if self.api_keys.is_empty() {
                    "Press Enter to open manager".to_string()
                } else {
                    format!("{} key(s) — Enter to manage", self.api_keys.len())
                }
            }
        }
    }

    pub fn field_hint(&self, field: &ConfigField) -> &'static str {
        match field {
            ConfigField::ModelSelection => "Up/Down to select from local models",
            ConfigField::Host => "0.0.0.0 for all interfaces, 127.0.0.1 for local",
            ConfigField::Port => "Port number (e.g. 8080)",
            ConfigField::MaxConcurrent => "Max simultaneous inference requests",
            ConfigField::RateLimit => "Token bucket: requests per second per IP",
            ConfigField::TimeoutSecs => "Kill stuck requests after N seconds",
            ConfigField::MaxBodySize => "Reject request bodies larger than N MB",
            ConfigField::CorsEnabled => "Allow cross-origin requests",
            ConfigField::MetricsEnabled => "Expose /metrics endpoint (Prometheus)",
            ConfigField::CompressionEnabled => "gzip response compression",
            ConfigField::QueueVisibility => "X-Queue-Position headers in responses",
            ConfigField::Backend => "auto, llama.cpp, or vllm",
            ConfigField::GpuRuntime => "auto, cuda, rocm, vulkan, metal, or cpu",
            ConfigField::GpuDevice => "GPU index (0, 1, 2, ...) or 'auto' for default",
            ConfigField::GpuLayers => {
                "-1 = all layers on GPU (CUDA/ROCm/Vulkan/Metal), 0 = CPU only"
            }
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
            ConfigField::ParallelSlots => "Parallel decoding slots (1=safe, 4=fast but more RAM)",
            ConfigField::LoraPaths => "Comma-separated paths to .gguf LoRA adapter files",
            ConfigField::DraftModel => {
                "Draft model for speculative decoding (path relative to models dir)"
            }
            ConfigField::DraftMaxTokens => "Max tokens draft model proposes per step (8-64)",
            ConfigField::DraftMinCtx => "Minimum context size for draft model",
            ConfigField::VectorStoreEnabled => "Enable integrated vector store for RAG",
            ConfigField::VectorStoreMaxDocs => "Max documents (0 = unlimited)",
            ConfigField::VectorStoreTopK => "Default search results count",
            ConfigField::OtelEnabled => "Enable OpenTelemetry distributed tracing",
            ConfigField::OtelEndpoint => "OTLP endpoint (e.g. http://localhost:4317)",
            ConfigField::OtelServiceName => "Service name for traces",
            ConfigField::OtelSampleRatio => "Sampling ratio 0.0-1.0",
            ConfigField::IpAllowlist => "Comma-separated IPs/CIDRs (empty = allow all)",
            ConfigField::IpDenylist => "Comma-separated IPs/CIDRs to block",
            ConfigField::StartServer => "Loads model and starts the API server",
            ConfigField::StopServer => "Stops the running server",
            ConfigField::LoadAdditionalModel => "Load another model while server is running",
            ConfigField::UnloadModel => "Unload a model from memory (Left/Right to pick)",
            ConfigField::SetDefaultModel => "Set which model handles requests without model field",
            ConfigField::ManageApiKeys => "Open the API key management modal",
        }
    }

    /// Raw value used to pre-fill the edit buffer. Unlike `field_value`,
    /// this never returns display placeholders ("(none)", "unlimited",
    /// masked secrets, ...) — those would otherwise be saved literally
    /// into the config when the user presses Enter.
    pub fn edit_value(&self, field: &ConfigField) -> String {
        match field {
            ConfigField::GpuDevice => match self.gpu_device {
                Some(d) => d.to_string(),
                None => String::new(),
            },
            ConfigField::GpuLayers => self.gpu_layers.to_string(),
            ConfigField::VectorStoreMaxDocs => self.vs_max_documents.to_string(),
            ConfigField::LoraPaths => self.lora_paths.clone(),
            ConfigField::DraftModel => self.draft_model.clone(),
            ConfigField::DraftMaxTokens => self.draft_max_tokens.to_string(),
            ConfigField::DraftMinCtx => self.draft_min_ctx.to_string(),
            ConfigField::OtelEndpoint => self.otel_endpoint.clone(),
            ConfigField::IpAllowlist => self.ip_allowlist.clone(),
            ConfigField::IpDenylist => self.ip_denylist.clone(),
            // Booleans are toggled, not edited — but keep a sane default.
            _ => self.field_value(field),
        }
    }

    pub fn start_edit(&mut self) {
        let field = self.current_field().clone();
        if !field.is_editable() {
            return;
        }
        self.edit_buffer = self.edit_value(&field);
        self.editing = true;
    }

    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
    }

    pub fn save_edit(&mut self) -> Result<(), String> {
        let field = self.current_field().clone();
        let value = self.edit_buffer.trim().to_string();

        match field {
            ConfigField::Host => {
                if value.is_empty() {
                    return Err("Host cannot be empty".to_string());
                }
                self.host = value;
            }
            ConfigField::Port => {
                let port = value
                    .parse::<u16>()
                    .map_err(|_| "Must be a valid port (1-65535)")?;
                if port == 0 {
                    return Err("Must be a valid port (1-65535)".to_string());
                }
                self.port = port;
            }
            ConfigField::MaxConcurrent => {
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
                if n == 0 {
                    return Err("Must be at least 1 (0 would block all requests)".to_string());
                }
                self.max_concurrent = n;
            }
            ConfigField::RateLimit => {
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
                if n == 0 {
                    return Err("Must be at least 1 (0 would block all requests)".to_string());
                }
                self.rate_limit = n;
            }
            ConfigField::TimeoutSecs => {
                self.timeout_secs = value
                    .parse::<u64>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::MaxBodySize => {
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
                if n == 0 {
                    return Err("Must be at least 1 MB".to_string());
                }
                self.max_body_size = n;
            }
            ConfigField::Backend => {
                self.backend = match value.to_lowercase().as_str() {
                    "auto" => BackendType::Auto,
                    "llama.cpp" | "llamacpp" | "llama" => BackendType::LlamaCpp,
                    "vllm" => BackendType::Vllm,
                    _ => return Err("Must be: auto, llama.cpp, or vllm".to_string()),
                };
            }
            ConfigField::GpuRuntime => {
                self.gpu_runtime = match value.to_lowercase().as_str() {
                    "auto" => athenas_core::GpuRuntime::Auto,
                    "cuda" => athenas_core::GpuRuntime::Cuda,
                    "rocm" => athenas_core::GpuRuntime::Rocm,
                    "vulkan" => athenas_core::GpuRuntime::Vulkan,
                    "metal" => athenas_core::GpuRuntime::Metal,
                    "cpu" => athenas_core::GpuRuntime::Cpu,
                    _ => return Err("Must be: auto, cuda, rocm, vulkan, metal, or cpu".to_string()),
                };
            }
            ConfigField::GpuDevice => {
                self.gpu_device =
                    if value.is_empty() || value.to_lowercase() == "auto" {
                        None
                    } else {
                        Some(value.parse::<u32>().map_err(|_| {
                            "Must be a GPU index (0, 1, 2, ...) or 'auto'".to_string()
                        })?)
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
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
                if n < 256 {
                    return Err("Must be at least 256 tokens".to_string());
                }
                self.context_size = n;
            }
            ConfigField::BatchSize => {
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
                if n == 0 {
                    return Err("Must be at least 1".to_string());
                }
                self.batch_size = n;
            }
            ConfigField::Threads => {
                self.threads = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a number (0 = auto)")?;
            }
            ConfigField::MaxTokens => {
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number")?;
                if n == 0 {
                    return Err("Must be at least 1".to_string());
                }
                self.max_tokens = n;
            }
            ConfigField::Temperature => {
                let v = value
                    .parse::<f32>()
                    .map_err(|_| "Must be a float (0.0-2.0)")?;
                if !(0.0..=2.0).contains(&v) {
                    return Err("Must be between 0.0 and 2.0".to_string());
                }
                self.temperature = v;
            }
            ConfigField::TopP => {
                let v = value
                    .parse::<f32>()
                    .map_err(|_| "Must be a float (0.0-1.0)")?;
                if !(0.0..=1.0).contains(&v) {
                    return Err("Must be between 0.0 and 1.0".to_string());
                }
                self.top_p = v;
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
            ConfigField::ParallelSlots => {
                let n = value.parse::<u32>().map_err(|_| "Must be 1-16")?;
                if !(1..=16).contains(&n) {
                    return Err("Must be 1-16".to_string());
                }
                self.parallel_slots = n;
            }
            ConfigField::LoraPaths => {
                self.lora_paths = value.to_string();
            }
            ConfigField::DraftModel => {
                self.draft_model = value.to_string();
            }
            ConfigField::DraftMaxTokens => {
                let n = value.parse::<u32>().map_err(|_| "Must be 1-256")?;
                if !(1..=256).contains(&n) {
                    return Err("Must be 1-256".to_string());
                }
                self.draft_max_tokens = n;
            }
            ConfigField::DraftMinCtx => {
                let n = value.parse::<u32>().map_err(|_| "Must be 64-65536")?;
                if !(64..=65536).contains(&n) {
                    return Err("Must be 64-65536".to_string());
                }
                self.draft_min_ctx = n;
            }
            ConfigField::VectorStoreMaxDocs => {
                self.vs_max_documents = value
                    .parse::<usize>()
                    .map_err(|_| "0 = unlimited, N = limit")?;
            }
            ConfigField::VectorStoreTopK => {
                self.vs_top_k = value
                    .parse::<usize>()
                    .map_err(|_| "Must be a positive number")?;
            }
            ConfigField::OtelEndpoint => {
                self.otel_endpoint = value.to_string();
            }
            ConfigField::OtelServiceName => {
                self.otel_service_name = value.to_string();
            }
            ConfigField::OtelSampleRatio => {
                let v = value.parse::<f64>().map_err(|_| "0.0-1.0")?;
                if !(0.0..=1.0).contains(&v) {
                    return Err("Must be between 0.0 and 1.0".to_string());
                }
                self.otel_sample_ratio = v;
            }
            ConfigField::IpAllowlist => {
                self.ip_allowlist = value.to_string();
            }
            ConfigField::IpDenylist => {
                self.ip_denylist = value.to_string();
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
            ConfigField::QueueVisibility => {
                self.queue_visibility = !self.queue_visibility;
            }
            ConfigField::FlashAttention => self.flash_attention = !self.flash_attention,
            ConfigField::Reasoning => self.reasoning_enabled = !self.reasoning_enabled,
            ConfigField::AutoResourceLimits => {
                self.auto_resource_limits = !self.auto_resource_limits;
            }
            ConfigField::VectorStoreEnabled => self.vs_enabled = !self.vs_enabled,
            ConfigField::OtelEnabled => self.otel_enabled = !self.otel_enabled,
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

    /// Estimate the RAM/VRAM footprint of loading the currently selected
    /// model with the current panel settings. Returns None when no model
    /// is selected. This is a heuristic — the same one used by the chat
    /// loader: file size + context/KV-cache overhead.
    pub fn estimate_selected_model_load(&self) -> Option<LoadEstimate> {
        let model = self.models.get(self.model_selected)?;
        let model_size_mb = model.file_size_bytes / (1024 * 1024);
        let full_ram = estimate_model_ram_mb(model_size_mb, self.context_size);
        let ctx_overhead = full_ram - model_size_mb;

        let ram_available_mb = self.hardware.memory_available_mb;
        let gpu_offload = self.gpu_layers != 0 && !self.hardware.gpus.is_empty();
        let full_gpu_offload = gpu_offload && self.gpu_layers < 0;
        let partial_gpu_offload = gpu_offload && self.gpu_layers > 0;

        let target_gpu = if gpu_offload {
            self.hardware
                .gpus
                .iter()
                .find(|g| Some(g.index) == self.gpu_device)
                .or_else(|| self.hardware.gpus.first())
        } else {
            None
        };

        // For APUs (integrated GPUs with unified memory), the "VRAM" reported
        // by the driver is just a small dedicated portion (e.g. 512 MB).
        // The GPU actually uses system RAM, so we should check against
        // system RAM available, not the tiny dedicated VRAM.
        let is_apu = target_gpu.map(|g| g.is_apu).unwrap_or(false);
        let vram_free_mb = target_gpu.map(|g| g.vram_total_mb.saturating_sub(g.vram_used_mb));

        let (ram_mb, vram_mb) = if full_gpu_offload {
            // Weights + KV cache go to VRAM; host keeps runtime buffers only
            (512 + ctx_overhead / 4, Some(model_size_mb + ctx_overhead))
        } else if partial_gpu_offload {
            // Unknown layer split — RAM shown is a safe upper bound,
            // VRAM a rough midpoint.
            (full_ram, Some(model_size_mb / 2))
        } else {
            (full_ram, None)
        };

        // Unknown available memory (detection failed) → don't claim it won't fit
        let fits_ram = ram_available_mb == 0 || ram_mb <= ram_available_mb;
        // For APUs, check against system RAM instead of dedicated VRAM,
        // since the GPU uses unified memory (shared system RAM).
        let fits_vram = if is_apu {
            // APU uses system RAM — check if the total (RAM + VRAM portion)
            // fits in available system memory
            ram_available_mb == 0 || vram_mb.unwrap_or(0) <= ram_available_mb
        } else {
            match (vram_mb, vram_free_mb) {
                (Some(need), Some(free)) if free > 0 => need <= free,
                _ => true,
            }
        };
        let fits = fits_ram && fits_vram;
        let tight = fits && ram_available_mb > 0 && ram_mb * 10 >= ram_available_mb * 9;

        Some(LoadEstimate {
            model_size_mb,
            ram_mb,
            vram_mb,
            ram_available_mb,
            vram_free_mb,
            full_gpu_offload,
            partial_gpu_offload,
            gpu_layers: self.gpu_layers,
            fits,
            tight,
        })
    }

    /// Returns the first active multi-tenant API key to use for TUI→server auth.
    /// Returns None when no keys exist (bootstrap mode — server allows no-auth).
    pub fn auth_bearer(&self) -> Option<&str> {
        self.api_keys
            .iter()
            .find(|k| k.active)
            .map(|k| k.api_key.as_str())
    }

    pub fn api_key_select_next(&mut self) {
        if !self.api_keys.is_empty() {
            self.api_key_selected = (self.api_key_selected + 1) % self.api_keys.len();
        }
    }

    pub fn api_key_select_prev(&mut self) {
        if !self.api_keys.is_empty() {
            if self.api_key_selected == 0 {
                self.api_key_selected = self.api_keys.len() - 1;
            } else {
                self.api_key_selected -= 1;
            }
        }
    }

    pub fn selected_api_key(&self) -> Option<&ApiKeyInfo> {
        self.api_keys.get(
            self.api_key_selected
                .min(self.api_keys.len().saturating_sub(1)),
        )
    }

    pub fn build_load_config(&self, model_path: &str) -> ModelLoadConfig {
        ModelLoadConfig {
            model_path: model_path.to_string(),
            gpu_layers: self.gpu_layers,
            gpu_runtime: self.gpu_runtime,
            gpu_device: self.gpu_device,
            context_size: self.context_size,
            batch_size: self.batch_size,
            threads: self.threads,
            flash_attention: self.flash_attention,
            use_mmap: true,
            use_mlock: false,
            reasoning_enabled: self.reasoning_enabled,
            reasoning_budget: self.reasoning_budget,
            mmproj_path: None,
            lora_paths: if self.lora_paths.is_empty() {
                Vec::new()
            } else {
                self.lora_paths
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            },
            parallel_slots: self.parallel_slots,
            draft_model_path: if self.draft_model.is_empty() {
                None
            } else {
                Some(self.draft_model.clone())
            },
            draft_max_tokens: self.draft_max_tokens,
            draft_min_ctx: self.draft_min_ctx,
        }
    }

    pub fn build_app_config(&self, base: &AppConfig) -> AppConfig {
        let mut config = base.clone();
        config.server.default_host = self.host.clone();
        config.server.default_port = self.port;
        config.server.max_concurrent_requests = self.max_concurrent;
        config.server.rate_limit_per_second = self.rate_limit;
        config.server.request_timeout_secs = self.timeout_secs;
        config.server.max_body_size_mb = self.max_body_size;
        config.server.cors_enabled = self.cors_enabled;
        config.server.enable_metrics = self.metrics_enabled;
        config.server.enable_compression = self.compression_enabled;
        config.server.queue_visibility = self.queue_visibility;
        config.inference.default_backend = self.backend;
        config.inference.default_gpu_layers = self.gpu_layers;
        config.inference.gpu_runtime = self.gpu_runtime;
        config.inference.gpu_device = self.gpu_device;
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
        config.inference.lora_paths = if self.lora_paths.is_empty() {
            Vec::new()
        } else {
            self.lora_paths
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };
        config.inference.parallel_slots = self.parallel_slots;
        config.inference.draft_model = if self.draft_model.is_empty() {
            None
        } else {
            Some(self.draft_model.clone())
        };
        config.inference.draft_max_tokens = self.draft_max_tokens;
        config.inference.draft_min_ctx = self.draft_min_ctx;
        config.server.vector_store.enabled = self.vs_enabled;
        config.server.vector_store.max_documents = self.vs_max_documents;
        config.server.vector_store.default_top_k = self.vs_top_k;
        config.server.otel.enabled = self.otel_enabled;
        config.server.otel.endpoint = if self.otel_endpoint.is_empty() {
            None
        } else {
            Some(self.otel_endpoint.clone())
        };
        config.server.otel.service_name = self.otel_service_name.clone();
        config.server.otel.sample_ratio = self.otel_sample_ratio;
        config.server.ip_allowlist = if self.ip_allowlist.is_empty() {
            Vec::new()
        } else {
            self.ip_allowlist
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };
        config.server.ip_denylist = if self.ip_denylist.is_empty() {
            Vec::new()
        } else {
            self.ip_denylist
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use athenas_core::{ModelFormat, ModelInfo};

    fn test_hardware() -> HardwareInfo {
        HardwareInfo {
            cpus: 8,
            memory_total_mb: 16 * 1024,
            memory_available_mb: 10 * 1024,
            gpus: Vec::new(),
            has_cuda: false,
            has_rocm: false,
            has_vulkan: false,
            has_metal: false,
        }
    }

    fn test_model(size_mb: u64) -> ModelInfo {
        ModelInfo {
            id: "test/model.gguf".to_string(),
            repo_id: "test".to_string(),
            name: "model.gguf".to_string(),
            format: ModelFormat::Gguf,
            file_path: std::path::PathBuf::from("/tmp/model.gguf"),
            file_size_bytes: size_mb * 1024 * 1024,
            quantization: Some("Q4_K_M".to_string()),
            context_length: None,
            architecture: None,
            huggingface_url: None,
            license: None,
            tags: Vec::new(),
            downloaded_at: chrono::Utc::now(),
            last_used_at: None,
            category: None,
        }
    }

    fn test_state() -> ServerPanelState {
        let mut config = AppConfig::default();
        // Point at a nonexistent dir so no real models are scanned
        config.paths.models_dir = std::path::PathBuf::from("/nonexistent");
        ServerPanelState::new(&config, test_hardware())
    }

    #[test]
    fn edit_value_never_returns_placeholders() {
        let state = test_state();
        assert_eq!(state.edit_value(&ConfigField::IpAllowlist), "");
        assert_eq!(state.edit_value(&ConfigField::IpDenylist), "");
        assert_eq!(state.edit_value(&ConfigField::LoraPaths), "");
        assert_eq!(state.edit_value(&ConfigField::OtelEndpoint), "");
        assert_eq!(state.edit_value(&ConfigField::GpuDevice), "");
        // "unlimited" display value must become the raw number
        assert_ne!(
            state.edit_value(&ConfigField::VectorStoreMaxDocs),
            "unlimited"
        );
    }

    #[test]
    fn save_edit_validates_ranges() {
        let mut state = test_state();

        // Port 0 is invalid
        state.selected = state
            .fields
            .iter()
            .position(|f| *f == ConfigField::Port)
            .unwrap();
        state.edit_buffer = "0".to_string();
        assert!(state.save_edit().is_err());
        state.edit_buffer = "8080".to_string();
        assert!(state.save_edit().is_ok());
        assert_eq!(state.port, 8080);

        // Temperature out of range
        state.selected = state
            .fields
            .iter()
            .position(|f| *f == ConfigField::Temperature)
            .unwrap();
        state.edit_buffer = "3.5".to_string();
        assert!(state.save_edit().is_err());
        state.edit_buffer = "0.7".to_string();
        assert!(state.save_edit().is_ok());

        // Parallel slots out of range
        state.selected = state
            .fields
            .iter()
            .position(|f| *f == ConfigField::ParallelSlots)
            .unwrap();
        state.edit_buffer = "99".to_string();
        assert!(state.save_edit().is_err());

        // Zero rate limit / max concurrent would block all requests
        state.selected = state
            .fields
            .iter()
            .position(|f| *f == ConfigField::RateLimit)
            .unwrap();
        state.edit_buffer = "0".to_string();
        assert!(state.save_edit().is_err());
    }

    #[test]
    fn estimate_cpu_only_matches_shared_heuristic() {
        let mut state = test_state();
        state.models = vec![test_model(4096)];
        state.model_selected = 0;
        state.gpu_layers = 0;
        state.context_size = 4096;

        let est = state.estimate_selected_model_load().unwrap();
        assert_eq!(est.model_size_mb, 4096);
        // 4096 ctx → (4096/1024)*64 = 256 MB overhead
        assert_eq!(est.ram_mb, 4096 + 256);
        assert_eq!(est.vram_mb, None);
        assert!(est.fits);
        assert!(!est.tight);
    }

    #[test]
    fn estimate_detects_when_model_does_not_fit() {
        let mut state = test_state();
        state.models = vec![test_model(12 * 1024)]; // 12 GB model, 10 GB free
        state.model_selected = 0;
        state.gpu_layers = 0;
        state.context_size = 4096;

        let est = state.estimate_selected_model_load().unwrap();
        assert!(!est.fits);
    }

    #[test]
    fn estimate_full_gpu_offload_moves_weight_to_vram() {
        let mut state = test_state();
        state.hardware.gpus.push(athenas_core::GpuInfo {
            index: 0,
            name: "Test GPU".to_string(),
            vendor: athenas_core::hardware::GpuVendor::Nvidia,
            vram_total_mb: 12 * 1024,
            vram_used_mb: 0,
            driver_version: "test".to_string(),
            compute_capability: None,
            is_apu: false,
        });
        state.models = vec![test_model(4096)];
        state.model_selected = 0;
        state.gpu_layers = -1;
        state.context_size = 4096;

        let est = state.estimate_selected_model_load().unwrap();
        assert!(est.full_gpu_offload);
        assert_eq!(est.vram_mb, Some(4096 + 256));
        assert!(est.ram_mb < est.model_size_mb); // host keeps only overhead
        assert!(est.fits);
    }

    #[test]
    fn refresh_models_clamps_selection() {
        let mut state = test_state();
        state.models = vec![test_model(10)];
        state.model_selected = 5; // out of bounds
        let config = AppConfig::default();
        state.refresh_models(&config); // real dir scan → 0 or N models, index must be valid
        assert!(state.model_selected < state.models.len().max(1));
    }
}
