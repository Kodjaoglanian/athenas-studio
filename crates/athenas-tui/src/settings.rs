use athenas_core::AppConfig;

#[derive(Clone, PartialEq)]
pub enum SettingsField {
    HfToken,
    Backend,
    GpuLayers,
    ContextSize,
    Temperature,
    MaxTokens,
    FlashAttention,
    ServerHost,
    ServerPort,
    ServerApiKey,
}

impl SettingsField {
    pub fn all() -> Vec<SettingsField> {
        vec![
            SettingsField::HfToken,
            SettingsField::Backend,
            SettingsField::GpuLayers,
            SettingsField::ContextSize,
            SettingsField::Temperature,
            SettingsField::MaxTokens,
            SettingsField::FlashAttention,
            SettingsField::ServerHost,
            SettingsField::ServerPort,
            SettingsField::ServerApiKey,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            SettingsField::HfToken => "HF Token",
            SettingsField::Backend => "Backend",
            SettingsField::GpuLayers => "GPU Layers",
            SettingsField::ContextSize => "Context Size",
            SettingsField::Temperature => "Temperature",
            SettingsField::MaxTokens => "Max Tokens",
            SettingsField::FlashAttention => "Flash Attention",
            SettingsField::ServerHost => "Server Host",
            SettingsField::ServerPort => "Server Port",
            SettingsField::ServerApiKey => "Server API Key",
        }
    }

    pub fn section(&self) -> &'static str {
        match self {
            SettingsField::HfToken => "HuggingFace",
            SettingsField::Backend
            | SettingsField::GpuLayers
            | SettingsField::ContextSize
            | SettingsField::Temperature
            | SettingsField::MaxTokens
            | SettingsField::FlashAttention => "Inference",
            SettingsField::ServerHost | SettingsField::ServerPort | SettingsField::ServerApiKey => {
                "Server"
            }
        }
    }
}

pub struct SettingsState {
    pub fields: Vec<SettingsField>,
    pub selected: usize,
    pub editing: bool,
    pub edit_buffer: String,
    pub config: AppConfig,
    pub status_message: Option<String>,
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
        }
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

    pub fn start_edit(&mut self) {
        let field = self.current_field().clone();
        self.edit_buffer = self.field_value(&field);
        if field == SettingsField::HfToken && !self.edit_buffer.is_empty() {
            self.edit_buffer.clear();
            self.edit_buffer.push_str("[hidden — type to replace]");
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

        if value == "[hidden — type to replace]" {
            self.editing = false;
            self.edit_buffer.clear();
            return Ok(());
        }

        match field {
            SettingsField::HfToken => {
                self.config.huggingface.token = if value.is_empty() { None } else { Some(value) };
            }
            SettingsField::Backend => {
                self.config.inference.default_backend = match value.as_str() {
                    "llama.cpp" | "llamacpp" => athenas_core::BackendType::LlamaCpp,
                    "vllm" => athenas_core::BackendType::Vllm,
                    "auto" => athenas_core::BackendType::Auto,
                    _ => return Err("Use: llama.cpp, vllm, or auto".to_string()),
                };
            }
            SettingsField::GpuLayers => {
                self.config.inference.default_gpu_layers = value
                    .parse()
                    .map_err(|_| "Must be a number (-1 for all)".to_string())?;
            }
            SettingsField::ContextSize => {
                self.config.inference.default_context_size =
                    value.parse().map_err(|_| "Must be a number".to_string())?;
            }
            SettingsField::Temperature => {
                self.config.inference.default_temperature = value
                    .parse()
                    .map_err(|_| "Must be a float (0.0-2.0)".to_string())?;
            }
            SettingsField::MaxTokens => {
                self.config.inference.default_max_tokens =
                    value.parse().map_err(|_| "Must be a number".to_string())?;
            }
            SettingsField::FlashAttention => {
                self.config.inference.flash_attention = value == "true" || value == "1";
            }
            SettingsField::ServerHost => {
                self.config.server.default_host = value;
            }
            SettingsField::ServerPort => {
                self.config.server.default_port = value
                    .parse()
                    .map_err(|_| "Must be a valid port (1-65535)".to_string())?;
            }
            SettingsField::ServerApiKey => {
                self.config.server.api_key = if value.is_empty() { None } else { Some(value) };
            }
        }

        self.config.save().map_err(|e| e.to_string())?;

        self.editing = false;
        self.edit_buffer.clear();
        self.status_message = Some("Saved!".to_string());
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
                    String::new()
                }
            }
            SettingsField::Backend => self.config.inference.default_backend.to_string(),
            SettingsField::GpuLayers => self.config.inference.default_gpu_layers.to_string(),
            SettingsField::ContextSize => self.config.inference.default_context_size.to_string(),
            SettingsField::Temperature => self.config.inference.default_temperature.to_string(),
            SettingsField::MaxTokens => self.config.inference.default_max_tokens.to_string(),
            SettingsField::FlashAttention => {
                if self.config.inference.flash_attention {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            SettingsField::ServerHost => self.config.server.default_host.clone(),
            SettingsField::ServerPort => self.config.server.default_port.to_string(),
            SettingsField::ServerApiKey => {
                if self.config.server.api_key.is_some() {
                    "[set]".to_string()
                } else {
                    String::new()
                }
            }
        }
    }

    pub fn field_hint(&self, field: &SettingsField) -> &'static str {
        match field {
            SettingsField::HfToken => "Get token at huggingface.co/settings/tokens",
            SettingsField::Backend => "llama.cpp | vllm | auto",
            SettingsField::GpuLayers => "-1 for all GPU layers",
            SettingsField::ContextSize => "e.g. 4096",
            SettingsField::Temperature => "0.0 - 2.0",
            SettingsField::MaxTokens => "e.g. 2048",
            SettingsField::FlashAttention => "true | false",
            SettingsField::ServerHost => "e.g. 127.0.0.1",
            SettingsField::ServerPort => "e.g. 8080",
            SettingsField::ServerApiKey => "Leave empty to disable auth",
        }
    }
}
