#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub input_text: String,
    pub scroll: usize,
    pub is_generating: bool,
    pub current_model: Option<String>,
    pub current_backend: Option<String>,
    pub tokens_per_second: Option<f32>,
    pub streaming_text: String,
    pub generation_start: Option<std::time::Instant>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: vec![ChatMessage {
                role: "system".to_string(),
                content: "Welcome to Athenas Studio!\n\n  F1: Chat | F2: Models | F3: Browser | F4: Settings\n  Type /help for commands. Press Ctrl+C to quit.".to_string(),
            }],
            input_text: String::new(),
            scroll: 0,
            is_generating: false,
            current_model: None,
            current_backend: None,
            tokens_per_second: None,
            streaming_text: String::new(),
            generation_start: None,
        }
    }
}

impl ChatState {
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.streaming_text.clear();
        self.is_generating = false;
        self.generation_start = None;
    }

    pub fn append_streaming(&mut self, text: &str) {
        self.streaming_text.push_str(text);
    }

    pub fn finalize_streaming(&mut self) {
        if !self.streaming_text.is_empty() {
            self.add_message("assistant", &self.streaming_text.clone());
            self.streaming_text.clear();
        }
        self.is_generating = false;
        self.generation_start = None;
    }
}
