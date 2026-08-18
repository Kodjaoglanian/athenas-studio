#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Reasoning/thinking tokens (shown collapsible in TUI)
    pub reasoning: String,
    /// Whether the reasoning section is expanded or collapsed
    pub reasoning_expanded: bool,
    /// Unix timestamp (seconds) when the message was created
    pub timestamp: u64,
}

impl ChatMessage {
    /// Format the timestamp as HH:MM
    pub fn time_str(&self) -> String {
        if self.timestamp == 0 {
            return String::new();
        }
        let secs = self.timestamp;
        let h = (secs / 3600) % 24;
        let m = (secs / 60) % 60;
        format!("{:02}:{:02}", h, m)
    }
}

fn now_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub input_text: String,
    pub scroll: usize,
    /// Maximum scroll position (set by render, used by key handler)
    pub max_scroll: usize,
    pub is_generating: bool,
    pub current_model: Option<String>,
    pub current_backend: Option<String>,
    pub tokens_per_second: Option<f32>,
    pub streaming_text: String,
    pub streaming_reasoning: String,
    pub generation_start: Option<std::time::Instant>,
    /// When true, scroll follows the latest content automatically
    pub auto_scroll: bool,
    /// GPU info string for status bar (e.g. "RTX 3090 (24000MB)")
    pub gpu_info: String,
    /// GPU runtime string for status bar (e.g. "cuda")
    pub gpu_runtime: String,
    /// Number of GPU layers (e.g. -1, 0, 20)
    pub gpu_layers: i32,
    /// Custom system prompt sent to the model (set via /system command).
    /// Empty string = no custom system prompt.
    pub system_prompt: String,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: vec![ChatMessage {
                role: "system".to_string(),
                content: "Welcome to Athenas Studio!\n\n  F1: Chat | F2: Models | F3: Browser | F4: Settings\n  Type /help for commands. Press Ctrl+C to quit.".to_string(),
                reasoning: String::new(),
                reasoning_expanded: false,
                timestamp: now_timestamp(),
            }],
            input_text: String::new(),
            scroll: 0,
            max_scroll: 0,
            is_generating: false,
            current_model: None,
            current_backend: None,
            tokens_per_second: None,
            streaming_text: String::new(),
            streaming_reasoning: String::new(),
            generation_start: None,
            auto_scroll: true,
            gpu_info: String::new(),
            gpu_runtime: String::new(),
            gpu_layers: -1,
            system_prompt: String::new(),
        }
    }
}

impl ChatState {
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            reasoning: String::new(),
            reasoning_expanded: false,
            timestamp: now_timestamp(),
        });
    }

    pub fn add_assistant_message(&mut self, content: &str, reasoning: &str) {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
            reasoning: reasoning.to_string(),
            reasoning_expanded: false,
            timestamp: now_timestamp(),
        });
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.streaming_text.clear();
        self.streaming_reasoning.clear();
        self.is_generating = false;
        self.generation_start = None;
    }

    pub fn append_streaming(&mut self, text: &str) {
        self.streaming_text.push_str(text);
    }

    pub fn append_reasoning(&mut self, text: &str) {
        self.streaming_reasoning.push_str(text);
    }

    pub fn finalize_streaming(&mut self) {
        if !self.streaming_text.is_empty() || !self.streaming_reasoning.is_empty() {
            let content = if self.streaming_text.is_empty() && !self.streaming_reasoning.is_empty()
            {
                "(Model produced only thinking/reasoning but no response. \
                 Try rephrasing, increasing max_tokens, or disabling reasoning in Settings.)"
                    .to_string()
            } else {
                self.streaming_text.clone()
            };
            self.messages.push(ChatMessage {
                role: "assistant".to_string(),
                content,
                reasoning: self.streaming_reasoning.clone(),
                reasoning_expanded: false,
                timestamp: now_timestamp(),
            });
            self.streaming_text.clear();
            self.streaming_reasoning.clear();
        }
        self.is_generating = false;
        self.generation_start = None;
    }
}
