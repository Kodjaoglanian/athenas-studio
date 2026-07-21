

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
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: vec![ChatMessage {
                role: "system".to_string(),
                content: "Welcome to Athenas Studio! Type a message to start chatting.".to_string(),
            }],
            input_text: String::new(),
            scroll: 0,
            is_generating: false,
            current_model: None,
            current_backend: None,
            tokens_per_second: None,
            streaming_text: String::new(),
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
    }
}
