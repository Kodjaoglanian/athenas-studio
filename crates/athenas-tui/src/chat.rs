#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Reasoning/thinking tokens (shown collapsible in TUI)
    pub reasoning: String,
    /// Whether the reasoning section is expanded or collapsed
    pub reasoning_expanded: bool,
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
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: vec![ChatMessage {
                role: "system".to_string(),
                content: "Welcome to Athenas Studio!\n\n  F1: Chat | F2: Models | F3: Browser | F4: Settings\n  Type /help for commands. Press Ctrl+C to quit.".to_string(),
                reasoning: String::new(),
                reasoning_expanded: false,
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
        });
    }

    pub fn add_assistant_message(&mut self, content: &str, reasoning: &str) {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
            reasoning: reasoning.to_string(),
            reasoning_expanded: false,
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
            });
            self.streaming_text.clear();
            self.streaming_reasoning.clear();
        }
        self.is_generating = false;
        self.generation_start = None;
    }
}
