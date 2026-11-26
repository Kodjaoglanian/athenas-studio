use std::collections::HashMap;

use chrono::{Local, TimeZone};
use ratatui::text::Line;

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
    /// Format the timestamp as local HH:MM (consistent with Logs).
    pub fn time_str(&self) -> String {
        if self.timestamp == 0 {
            return String::new();
        }
        Local
            .timestamp_opt(self.timestamp as i64, 0)
            .single()
            .map(|dt| dt.format("%H:%M").to_string())
            .unwrap_or_default()
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
    /// Cached rendered markdown per (message_index, wrap_width) so streaming
    /// does not re-parse the entire conversation every 100ms.
    pub md_cache: HashMap<(usize, usize), Vec<Line<'static>>>,
    /// Prompt history for Alt+Up / Alt+Down recall
    pub prompt_history: Vec<String>,
    pub history_index: Option<usize>,
    pub history_draft: String,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
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
            md_cache: HashMap::new(),
            prompt_history: Vec::new(),
            history_index: None,
            history_draft: String::new(),
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

    pub fn add_error_message(&mut self, content: &str) {
        self.add_message("error", content);
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
        self.md_cache.clear();
        self.streaming_text.clear();
        self.streaming_reasoning.clear();
        self.is_generating = false;
        self.generation_start = None;
        self.scroll = 0;
        self.max_scroll = 0;
        self.auto_scroll = true;
    }

    pub fn invalidate_md_cache(&mut self, idx: usize) {
        self.md_cache.retain(|&(m_idx, _), _| m_idx != idx);
    }

    /// Return cached rendered markdown lines for message `idx` at `width`.
    pub fn cached_markdown(&mut self, idx: usize, width: usize) -> Vec<Line<'static>> {
        if let Some(lines) = self.md_cache.get(&(idx, width)) {
            return lines.clone();
        }
        let content = self
            .messages
            .get(idx)
            .map(|m| m.content.as_str())
            .unwrap_or("");
        let rendered = crate::markdown::render_markdown(content, width);
        // Keep cache bounded on terminal resize
        if self.md_cache.len() > 256 {
            self.md_cache.clear();
        }
        self.md_cache.insert((idx, width), rendered.clone());
        rendered
    }

    pub fn push_history(&mut self, prompt: String) {
        if prompt.trim().is_empty() {
            return;
        }
        if self.prompt_history.last().map(|s| s.as_str()) != Some(prompt.as_str()) {
            self.prompt_history.push(prompt);
            if self.prompt_history.len() > 100 {
                self.prompt_history.remove(0);
            }
        }
        self.history_index = None;
        self.history_draft.clear();
    }

    pub fn history_prev(&mut self, current_input: &str) -> Option<String> {
        if self.prompt_history.is_empty() {
            return None;
        }
        let next_idx = match self.history_index {
            None => {
                self.history_draft = current_input.to_string();
                self.prompt_history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_index = Some(next_idx);
        self.prompt_history.get(next_idx).cloned()
    }

    pub fn history_next(&mut self) -> Option<String> {
        let idx = self.history_index?;
        if idx + 1 >= self.prompt_history.len() {
            self.history_index = None;
            Some(std::mem::take(&mut self.history_draft))
        } else {
            let next_idx = idx + 1;
            self.history_index = Some(next_idx);
            self.prompt_history.get(next_idx).cloned()
        }
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
