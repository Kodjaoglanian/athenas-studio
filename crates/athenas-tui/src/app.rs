use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;

use athenas_core::{AppConfig, HardwareInfo, ModelRegistry, Result};
use athenas_inference::{
    Backend, BackendFactory, ChatMessage, ChatRequest, ModelLoadConfig, Role, StreamChunk,
};

use crate::chat::ChatState;
use crate::components;
use crate::model_list::ModelListState;

pub enum AppMode {
    Chat,
    ModelList,
}

pub struct TuiApp {
    config: AppConfig,
    hardware: HardwareInfo,
    chat_state: ChatState,
    model_list_state: ModelListState,
    mode: AppMode,
    backend: Option<Box<dyn Backend>>,
}

impl TuiApp {
    pub fn new(config: AppConfig, hardware: HardwareInfo) -> Self {
        let registry = ModelRegistry::new(config.paths.models_dir.clone());
        let models = registry.list_local_models().unwrap_or_default();

        let mut model_list_state = ModelListState::default();
        model_list_state.set_models(models);

        Self {
            config,
            hardware,
            chat_state: ChatState::default(),
            model_list_state,
            mode: AppMode::Chat,
            backend: None,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        enable_raw_mode().map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)
            .map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal =
            Terminal::new(backend).map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;

        let result = self.main_loop(&mut terminal).await;

        disable_raw_mode().map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)
            .map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;
        terminal
            .show_cursor()
            .map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;

        result
    }

    async fn main_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> Result<()> {
        loop {
            terminal.draw(|f| self.render(f)).ok();

            if event::poll(std::time::Duration::from_millis(100))
                .map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?
            {
                let event =
                    event::read().map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;

                if let Event::Key(key) = event {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    // Global keys
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        break;
                    }

                    match self.mode {
                        AppMode::Chat => {
                            match key.code {
                                KeyCode::Enter => {
                                    self.send_message().await;
                                }
                                KeyCode::Char(c) => {
                                    self.chat_state.input_text.push(c);
                                }
                                KeyCode::Backspace => {
                                    self.chat_state.input_text.pop();
                                }
                                KeyCode::Tab => {
                                    self.mode = AppMode::ModelList;
                                }
                                KeyCode::Esc if self.chat_state.is_generating => {
                                    // Can't easily cancel, just ignore
                                }
                                _ => {}
                            }
                        }
                        AppMode::ModelList => match key.code {
                            KeyCode::Down | KeyCode::Char('j') => {
                                self.model_list_state.next();
                            }
                            KeyCode::Up | KeyCode::Char('k') => {
                                self.model_list_state.previous();
                            }
                            KeyCode::Enter => {
                                if let Some(path) = self
                                    .model_list_state
                                    .selected()
                                    .map(|m| m.file_path.to_string_lossy().to_string())
                                {
                                    self.load_model(&path).await;
                                    self.mode = AppMode::Chat;
                                }
                            }
                            KeyCode::Tab | KeyCode::Esc => {
                                self.mode = AppMode::Chat;
                            }
                            _ => {}
                        },
                    }
                }
            }

            // Check for streaming updates
            // (handled in send_message via tokio task)
        }

        Ok(())
    }

    fn render(&self, f: &mut ratatui::Frame) {
        let area = f.area();

        match self.mode {
            AppMode::Chat => {
                components::render_chat_area(f, area, &self.chat_state);
            }
            AppMode::ModelList => {
                components::render_model_list(f, area, &self.model_list_state);
            }
        }
    }

    async fn send_message(&mut self) {
        let text = self.chat_state.input_text.trim().to_string();
        if text.is_empty() {
            return;
        }

        // Handle commands
        if text.starts_with('/') {
            self.handle_command(&text).await;
            return;
        }

        if self.backend.is_none() {
            self.chat_state
                .add_message("system", "No model loaded. Press Tab to select a model.");
            return;
        }

        self.chat_state.add_message("user", &text);
        self.chat_state.input_text.clear();
        self.chat_state.is_generating = true;

        // Build chat request from current messages
        let messages: Vec<ChatMessage> = self
            .chat_state
            .messages
            .iter()
            .filter(|m| m.role != "system" || !m.content.contains("Welcome"))
            .map(|m| {
                let role = match m.role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "system" => Role::System,
                    _ => Role::User,
                };
                ChatMessage {
                    role,
                    content: m.content.clone(),
                }
            })
            .collect();

        let req = ChatRequest {
            model: String::new(),
            messages,
            temperature: Some(self.config.inference.default_temperature),
            top_p: Some(self.config.inference.default_top_p),
            max_tokens: Some(self.config.inference.default_max_tokens),
            stream: true,
            stop: None,
            seed: None,
        };

        // Stream response
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);

        // We need to take the backend out, use it, and put it back
        // Since Backend is not Clone, we'll use a different approach
        // For simplicity, we'll call chat_stream directly
        if let Some(backend) = &self.backend {
            let _ = backend.chat_stream(req, tx).await;
        }

        // Collect streamed chunks
        while let Some(chunk) = rx.recv().await {
            if chunk.done {
                if let Some(stats) = chunk.stats {
                    self.chat_state.tokens_per_second = Some(stats.tokens_per_second);
                }
                self.chat_state.finalize_streaming();
            } else {
                self.chat_state.append_streaming(&chunk.text);
            }
        }

        if self.chat_state.is_generating {
            self.chat_state.finalize_streaming();
        }
    }

    async fn handle_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        match parts[0] {
            "/clear" => {
                self.chat_state.clear();
            }
            "/model" => {
                self.mode = AppMode::ModelList;
            }
            "/help" => {
                self.chat_state
                    .add_message("system", "Commands: /clear, /model, /help, /quit");
            }
            "/quit" => {
                // Signal exit
                self.chat_state.add_message("system", "Use Ctrl+C to quit");
            }
            _ => {
                self.chat_state
                    .add_message("system", &format!("Unknown command: {}", parts[0]));
            }
        }
        self.chat_state.input_text.clear();
    }

    async fn load_model(&mut self, path: &str) {
        self.chat_state
            .add_message("system", &format!("Loading model: {}...", path));

        let mut backend =
            BackendFactory::create(self.config.inference.default_backend, &self.hardware)
                .unwrap_or_else(|e| {
                    self.chat_state
                        .add_message("system", &format!("Failed to create backend: {}", e));
                    panic!("Backend creation failed");
                });

        let load_config = ModelLoadConfig {
            model_path: path.to_string(),
            gpu_layers: self.config.inference.default_gpu_layers,
            context_size: self.config.inference.default_context_size,
            batch_size: self.config.inference.default_batch_size,
            threads: self.config.inference.default_threads,
            flash_attention: self.config.inference.flash_attention,
            use_mmap: true,
            use_mlock: false,
        };

        match backend.load_model(load_config).await {
            Ok(()) => {
                let info = backend.model_info();
                if let Some(ref i) = info {
                    self.chat_state.current_model = Some(i.name.clone());
                    self.chat_state.current_backend = Some(i.backend_name.clone());
                }
                self.chat_state
                    .add_message("system", "Model loaded successfully!");
                self.backend = Some(backend);
            }
            Err(e) => {
                self.chat_state
                    .add_message("system", &format!("Failed to load model: {}", e));
            }
        }
    }
}
