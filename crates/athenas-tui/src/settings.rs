use athenas_core::{AppConfig, BackendType, GpuRuntime};

fn parse_batch_size(value: &str) -> Result<u32, String> {
    let batch_size = value
        .parse::<u32>()
        .map_err(|_| "Must be a positive number".to_string())?;
    if batch_size == 0 {
        return Err("Must be a positive number".to_string());
    }
    Ok(batch_size)
}

#[derive(Clone, PartialEq)]
pub enum SettingsField {
    HfToken,
    Backend,
    Device,
    GpuRuntime,
    GpuDevice,
    GpuLayers,
    ContextSize,
    BatchSize,
    Threads,
    Temperature,
    TopP,
    MaxTokens,
    FlashAttention,
    Streaming,
    Reasoning,
    ReasoningBudget,
    RamReserve,
    CpuReserve,
    AutoResourceLimits,
    ServerHost,
    ServerPort,
}

impl SettingsField {
    pub fn all() -> Vec<SettingsField> {
        vec![
            SettingsField::HfToken,
            SettingsField::Backend,
            SettingsField::Device,
            SettingsField::GpuRuntime,
            SettingsField::GpuDevice,
            SettingsField::GpuLayers,
            SettingsField::ContextSize,
            SettingsField::BatchSize,
            SettingsField::Threads,
            SettingsField::Temperature,
            SettingsField::TopP,
            SettingsField::MaxTokens,
            SettingsField::FlashAttention,
            SettingsField::Streaming,
            SettingsField::Reasoning,
            SettingsField::ReasoningBudget,
            SettingsField::RamReserve,
            SettingsField::CpuReserve,
            SettingsField::AutoResourceLimits,
            SettingsField::ServerHost,
            SettingsField::ServerPort,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            SettingsField::HfToken => "HF Token",
            SettingsField::Backend => "Backend",
            SettingsField::Device => "Device",
            SettingsField::GpuRuntime => "GPU Runtime",
            SettingsField::GpuDevice => "GPU Device",
            SettingsField::GpuLayers => "GPU Layers",
            SettingsField::ContextSize => "Context Size",
            SettingsField::BatchSize => "Batch Size",
            SettingsField::Threads => "Threads",
            SettingsField::Temperature => "Temperature",
            SettingsField::TopP => "Top P",
            SettingsField::MaxTokens => "Max Tokens",
            SettingsField::FlashAttention => "Flash Attention",
            SettingsField::Streaming => "Streaming",
            SettingsField::Reasoning => "Reasoning/Thinking",
            SettingsField::ReasoningBudget => "Reasoning Budget",
            SettingsField::RamReserve => "RAM Reserve (MB)",
            SettingsField::CpuReserve => "CPU Reserve (cores)",
            SettingsField::AutoResourceLimits => "Auto Resource Limits",
            SettingsField::ServerHost => "Server Host",
            SettingsField::ServerPort => "Server Port",
        }
    }

    pub fn section(&self) -> &'static str {
        match self {
            SettingsField::HfToken => "HuggingFace",
            SettingsField::Backend
            | SettingsField::Device
            | SettingsField::GpuRuntime
            | SettingsField::GpuDevice
            | SettingsField::GpuLayers
            | SettingsField::ContextSize
            | SettingsField::BatchSize
            | SettingsField::Threads
            | SettingsField::Temperature
            | SettingsField::TopP
            | SettingsField::MaxTokens
            | SettingsField::FlashAttention
            | SettingsField::Streaming
            | SettingsField::Reasoning
            | SettingsField::ReasoningBudget
            | SettingsField::RamReserve
            | SettingsField::CpuReserve
            | SettingsField::AutoResourceLimits => "Inference",
            SettingsField::ServerHost | SettingsField::ServerPort => "Server",
        }
    }

    pub fn is_toggle(&self) -> bool {
        matches!(
            self,
            SettingsField::FlashAttention
                | SettingsField::Streaming
                | SettingsField::Reasoning
                | SettingsField::AutoResourceLimits
        )
    }

    pub fn is_cycle(&self) -> bool {
        matches!(
            self,
            SettingsField::Backend | SettingsField::Device | SettingsField::GpuRuntime
        )
    }
}

pub struct SettingsState {
    pub fields: Vec<SettingsField>,
    pub selected: usize,
    pub editing: bool,
    pub edit_buffer: String,
    pub config: AppConfig,
    pub status_message: Option<String>,
    pub status_is_error: bool,
    /// Detected hardware info for display
    pub hardware: Option<athenas_core::HardwareInfo>,
}

impl SettingsState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            fields: SettingsField::all(),
            selected: 0,
            editing: false,
            edit_buffer: String::new(),
            config,
            status_message: None,
            status_is_error: false,
            hardware: None,
        }
    }

    pub fn with_hardware(mut self, hw: athenas_core::HardwareInfo) -> Self {
        self.hardware = Some(hw);
        self
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
        self.status_is_error = false;
    }

    pub fn set_error(&mut self, msg: impl Into<String>) {
        self.status_message = Some(msg.into());
        self.status_is_error = true;
    }

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

    pub fn current_field(&self) -> &SettingsField {
        &self.fields[self.selected]
    }

    /// Raw value for pre-filling the edit buffer (no display placeholders).
    pub fn edit_value(&self, field: &SettingsField) -> String {
        match field {
            SettingsField::HfToken => String::new(),
            SettingsField::GpuDevice => match self.config.inference.gpu_device {
                Some(d) => d.to_string(),
                None => String::new(),
            },
            _ => self.field_value(field),
        }
    }

    pub fn start_edit(&mut self) {
        let field = self.current_field().clone();
        self.edit_buffer = self.edit_value(&field);
        self.editing = true;
    }

    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
    }

    /// Toggle a boolean field or cycle an enum field in-memory. Returns a human-readable summary.
    pub fn toggle_or_cycle_current(&mut self) -> Option<String> {
        let field = self.current_field().clone();
        match field {
            SettingsField::FlashAttention => {
                self.config.inference.flash_attention = !self.config.inference.flash_attention;
                Some(format!(
                    "Flash Attention: {}",
                    if self.config.inference.flash_attention {
                        "ON"
                    } else {
                        "OFF"
                    }
                ))
            }
            SettingsField::Streaming => {
                self.config.inference.streaming_enabled = !self.config.inference.streaming_enabled;
                Some(format!(
                    "Streaming: {}",
                    if self.config.inference.streaming_enabled {
                        "ON"
                    } else {
                        "OFF"
                    }
                ))
            }
            SettingsField::Reasoning => {
                self.config.inference.reasoning_enabled = !self.config.inference.reasoning_enabled;
                Some(format!(
                    "Reasoning: {}",
                    if self.config.inference.reasoning_enabled {
                        "ON"
                    } else {
                        "OFF"
                    }
                ))
            }
            SettingsField::AutoResourceLimits => {
                self.config.inference.auto_resource_limits =
                    !self.config.inference.auto_resource_limits;
                Some(format!(
                    "Auto Resource Limits: {}",
                    if self.config.inference.auto_resource_limits {
                        "ON"
                    } else {
                        "OFF"
                    }
                ))
            }
            SettingsField::Device => {
                let current = self.config.inference.default_gpu_layers;
                let new_val = if current == 0 { -1 } else { 0 };
                self.config.inference.default_gpu_layers = new_val;
                let device = if new_val == 0 { "CPU" } else { "GPU" };
                Some(format!("Device: {}", device))
            }
            SettingsField::Backend => {
                let next = match self.config.inference.default_backend {
                    BackendType::Auto => BackendType::LlamaCpp,
                    BackendType::LlamaCpp => BackendType::Vllm,
                    BackendType::Vllm => BackendType::Onnx,
                    BackendType::Onnx => BackendType::Auto,
                };
                self.config.inference.default_backend = next;
                Some(format!("Backend: {}", next))
            }
            SettingsField::GpuRuntime => {
                let next = match self.config.inference.gpu_runtime {
                    GpuRuntime::Auto => GpuRuntime::Cuda,
                    GpuRuntime::Cuda => GpuRuntime::Rocm,
                    GpuRuntime::Rocm => GpuRuntime::Vulkan,
                    GpuRuntime::Vulkan => GpuRuntime::Metal,
                    GpuRuntime::Metal => GpuRuntime::Cpu,
                    GpuRuntime::Cpu => GpuRuntime::Auto,
                };
                self.config.inference.gpu_runtime = next;
                Some(format!("GPU Runtime: {}", next))
            }
            _ => None,
        }
    }

    /// Validate and apply the current edit buffer to `self.config` in memory
    /// without blocking on disk I/O (caller persists via `spawn_blocking`).
    pub fn apply_edit(&mut self) -> Result<(), String> {
        let field = self.current_field().clone();
        let value = self.edit_buffer.trim().to_string();

        match field {
            SettingsField::HfToken => {
                self.config.huggingface.token = if value.is_empty() { None } else { Some(value) };
            }
            SettingsField::Backend => {
                self.config.inference.default_backend = match value.to_lowercase().as_str() {
                    "llama.cpp" | "llamacpp" => BackendType::LlamaCpp,
                    "vllm" => BackendType::Vllm,
                    "onnx" => BackendType::Onnx,
                    "auto" => BackendType::Auto,
                    _ => return Err("Use: llama.cpp, vllm, onnx, or auto".to_string()),
                };
            }
            SettingsField::Device => match value.to_lowercase().as_str() {
                "cpu" => {
                    self.config.inference.default_gpu_layers = 0;
                }
                "gpu" => {
                    if self.config.inference.default_gpu_layers == 0 {
                        self.config.inference.default_gpu_layers = -1;
                    }
                }
                _ => return Err("Use: cpu or gpu".to_string()),
            },
            SettingsField::GpuRuntime => {
                self.config.inference.gpu_runtime = match value.to_lowercase().as_str() {
                    "auto" => GpuRuntime::Auto,
                    "cuda" => GpuRuntime::Cuda,
                    "rocm" => GpuRuntime::Rocm,
                    "vulkan" => GpuRuntime::Vulkan,
                    "metal" => GpuRuntime::Metal,
                    "cpu" => GpuRuntime::Cpu,
                    _ => return Err("Use: auto, cuda, rocm, vulkan, metal, or cpu".to_string()),
                };
            }
            SettingsField::GpuDevice => {
                self.config.inference.gpu_device =
                    if value.is_empty() || value.to_lowercase() == "auto" {
                        None
                    } else {
                        Some(value.parse::<u32>().map_err(|_| {
                            "Must be a GPU index (0, 1, 2, ...) or 'auto'".to_string()
                        })?)
                    };
            }
            SettingsField::GpuLayers => {
                self.config.inference.default_gpu_layers = value
                    .parse()
                    .map_err(|_| "Must be a number (-1 for all)".to_string())?;
            }
            SettingsField::ContextSize => {
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number".to_string())?;
                if n < 256 {
                    return Err("Context Size must be at least 256".to_string());
                }
                self.config.inference.default_context_size = n;
            }
            SettingsField::BatchSize => {
                self.config.inference.default_batch_size = parse_batch_size(&value)?;
            }
            SettingsField::Threads => {
                self.config.inference.default_threads = value
                    .parse()
                    .map_err(|_| "Must be a number (0 = auto)".to_string())?;
            }
            SettingsField::Temperature => {
                let v = value
                    .parse::<f32>()
                    .map_err(|_| "Must be a float (0.0-2.0)".to_string())?;
                if !v.is_finite() || !(0.0..=2.0).contains(&v) {
                    return Err("Temperature must be between 0.0 and 2.0".to_string());
                }
                self.config.inference.default_temperature = v;
            }
            SettingsField::TopP => {
                let v = value
                    .parse::<f32>()
                    .map_err(|_| "Must be a float (0.0-1.0)".to_string())?;
                if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                    return Err("Top P must be between 0.0 and 1.0".to_string());
                }
                self.config.inference.default_top_p = v;
            }
            SettingsField::MaxTokens => {
                let n = value
                    .parse::<u32>()
                    .map_err(|_| "Must be a positive number".to_string())?;
                if n == 0 {
                    return Err("Max Tokens must be at least 1".to_string());
                }
                self.config.inference.default_max_tokens = n;
            }
            SettingsField::FlashAttention => {
                self.config.inference.flash_attention = value.eq_ignore_ascii_case("true")
                    || value == "1"
                    || value.eq_ignore_ascii_case("on");
            }
            SettingsField::Streaming => {
                self.config.inference.streaming_enabled = value.eq_ignore_ascii_case("true")
                    || value == "1"
                    || value.eq_ignore_ascii_case("on");
            }
            SettingsField::Reasoning => {
                self.config.inference.reasoning_enabled = value.eq_ignore_ascii_case("true")
                    || value == "1"
                    || value.eq_ignore_ascii_case("on");
            }
            SettingsField::ReasoningBudget => {
                self.config.inference.reasoning_budget = value
                    .parse()
                    .map_err(|_| "-1 = unlimited, 0 = disabled, N = token budget".to_string())?;
            }
            SettingsField::RamReserve => {
                self.config.inference.ram_reserve_mb = value
                    .parse()
                    .map_err(|_| "Must be a number (MB to reserve for OS)".to_string())?;
            }
            SettingsField::CpuReserve => {
                self.config.inference.cpu_reserve_cores = value
                    .parse()
                    .map_err(|_| "Must be a number (cores to leave free)".to_string())?;
            }
            SettingsField::AutoResourceLimits => {
                self.config.inference.auto_resource_limits = value.eq_ignore_ascii_case("true")
                    || value == "1"
                    || value.eq_ignore_ascii_case("on");
            }
            SettingsField::ServerHost => {
                if value.is_empty() {
                    return Err("Server Host cannot be empty".to_string());
                }
                self.config.server.default_host = value;
            }
            SettingsField::ServerPort => {
                let port = value
                    .parse::<u16>()
                    .map_err(|_| "Must be a valid port (1-65535)".to_string())?;
                if port == 0 {
                    return Err("Must be a valid port (1-65535)".to_string());
                }
                self.config.server.default_port = port;
            }
        }

        self.editing = false;
        self.edit_buffer.clear();
        Ok(())
    }

    pub fn save_edit(&mut self) -> Result<(), String> {
        self.apply_edit()?;
        self.config.save().map_err(|e| e.to_string())?;
        self.set_status("Saved!");
        Ok(())
    }

    pub fn field_value(&self, field: &SettingsField) -> String {
        match field {
            SettingsField::HfToken => {
                if self.config.huggingface.token.is_some() {
                    "[set]".to_string()
                } else if std::env::var("HF_TOKEN").is_ok() {
                    "[set via env]".to_string()
                } else {
                    "(not set)".to_string()
                }
            }
            SettingsField::Backend => self.config.inference.default_backend.to_string(),
            SettingsField::Device => {
                if self.config.inference.default_gpu_layers == 0 {
                    "CPU".to_string()
                } else {
                    "GPU".to_string()
                }
            }
            SettingsField::GpuRuntime => self.config.inference.gpu_runtime.to_string(),
            SettingsField::GpuDevice => match self.config.inference.gpu_device {
                Some(d) => d.to_string(),
                None => "auto".to_string(),
            },
            SettingsField::GpuLayers => {
                if self.config.inference.default_gpu_layers < 0 {
                    "all (-1)".to_string()
                } else {
                    self.config.inference.default_gpu_layers.to_string()
                }
            }
            SettingsField::ContextSize => self.config.inference.default_context_size.to_string(),
            SettingsField::BatchSize => self.config.inference.default_batch_size.to_string(),
            SettingsField::Threads => self.config.inference.default_threads.to_string(),
            SettingsField::Temperature => self.config.inference.default_temperature.to_string(),
            SettingsField::TopP => self.config.inference.default_top_p.to_string(),
            SettingsField::MaxTokens => self.config.inference.default_max_tokens.to_string(),
            SettingsField::FlashAttention => on_off(self.config.inference.flash_attention),
            SettingsField::Streaming => on_off(self.config.inference.streaming_enabled),
            SettingsField::Reasoning => on_off(self.config.inference.reasoning_enabled),
            SettingsField::ReasoningBudget => self.config.inference.reasoning_budget.to_string(),
            SettingsField::RamReserve => self.config.inference.ram_reserve_mb.to_string(),
            SettingsField::CpuReserve => self.config.inference.cpu_reserve_cores.to_string(),
            SettingsField::AutoResourceLimits => on_off(self.config.inference.auto_resource_limits),
            SettingsField::ServerHost => self.config.server.default_host.clone(),
            SettingsField::ServerPort => self.config.server.default_port.to_string(),
        }
    }

    pub fn field_hint(&self, field: &SettingsField) -> &'static str {
        match field {
            SettingsField::HfToken => {
                "Get token at huggingface.co/settings/tokens (leave empty to clear)"
            }
            SettingsField::Backend => "Enter/Space to cycle: auto | llama.cpp | vllm | onnx",
            SettingsField::Device => "Enter/Space to toggle: CPU | GPU",
            SettingsField::GpuRuntime => {
                "Enter/Space to cycle: auto | cuda | rocm | vulkan | metal | cpu"
            }
            SettingsField::GpuDevice => "GPU index (0, 1, 2, ...) or leave empty for auto",
            SettingsField::GpuLayers => "-1 = all GPU, 0 = CPU only, N = N layers",
            SettingsField::ContextSize => "Minimum 256 tokens (e.g. 4096, 8192)",
            SettingsField::BatchSize => "Positive integer (e.g. 512, 2048)",
            SettingsField::Threads => "0 = auto (uses CPU count - reserve)",
            SettingsField::Temperature => "0.0 - 2.0",
            SettingsField::TopP => "0.0 - 1.0 (nucleus sampling)",
            SettingsField::MaxTokens => "Minimum 1 (e.g. 2048)",
            SettingsField::FlashAttention => "Enter/Space to toggle ON/OFF",
            SettingsField::Streaming => "Enter/Space to toggle ON/OFF",
            SettingsField::Reasoning => {
                "Enter/Space to toggle ON/OFF (disable for Qwen3.5 if slow)"
            }
            SettingsField::ReasoningBudget => "-1 = unlimited, 0 = off, N = token limit",
            SettingsField::RamReserve => "MB reserved for OS (e.g. 2048)",
            SettingsField::CpuReserve => "Cores to leave free (e.g. 1)",
            SettingsField::AutoResourceLimits => "Enter/Space to toggle ON/OFF",
            SettingsField::ServerHost => "e.g. 127.0.0.1 or 0.0.0.0",
            SettingsField::ServerPort => "Valid TCP port 1 - 65535 (e.g. 8080)",
        }
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

    fn select_field(state: &mut SettingsState, field: SettingsField) {
        state.selected = state.fields.iter().position(|f| *f == field).unwrap();
    }

    #[test]
    fn batch_size_must_be_positive() {
        assert_eq!(parse_batch_size("2048"), Ok(2048));
        assert!(parse_batch_size("0").is_err());
        assert!(parse_batch_size("invalid").is_err());
    }

    #[test]
    fn settings_validation_ranges() {
        let mut state = SettingsState::new(AppConfig::default());

        // Temperature: 0.0..=2.0
        select_field(&mut state, SettingsField::Temperature);
        state.edit_buffer = "-0.1".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "2.5".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "NaN".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "0.7".to_string();
        assert!(state.apply_edit().is_ok());

        // TopP: 0.0..=1.0
        select_field(&mut state, SettingsField::TopP);
        state.edit_buffer = "1.5".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "-0.2".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "0.95".to_string();
        assert!(state.apply_edit().is_ok());

        // ContextSize: >= 256
        select_field(&mut state, SettingsField::ContextSize);
        state.edit_buffer = "0".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "128".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "4096".to_string();
        assert!(state.apply_edit().is_ok());

        // MaxTokens: >= 1
        select_field(&mut state, SettingsField::MaxTokens);
        state.edit_buffer = "0".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "2048".to_string();
        assert!(state.apply_edit().is_ok());

        // ServerPort: 1..=65535
        select_field(&mut state, SettingsField::ServerPort);
        state.edit_buffer = "0".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "70000".to_string();
        assert!(state.apply_edit().is_err());
        state.edit_buffer = "8080".to_string();
        assert!(state.apply_edit().is_ok());
    }
}
