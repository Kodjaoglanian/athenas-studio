use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;

use athenas_core::{AppConfig, HardwareInfo, ModelRegistry, Result};
use athenas_inference::{
    Backend, BackendFactory, ChatMessage, ChatRequest, MessageContent, ModelLoadConfig,
    RemoteBackend, Role, StreamChunk,
};

use crate::chat::ChatState;
use crate::components;
use crate::model_browser::{BrowserPhase, ModelBrowserState};
use crate::model_list::ModelListState;
use crate::server_manager;
use crate::server_panel::{ConfigField, ServerPanelState, ServerPhase};
use crate::settings::SettingsState;

enum AdditionalModelLoadResult {
    InProcess(std::result::Result<Box<dyn Backend>, (athenas_core::AthenasError, String, String)>),
    Detached(std::result::Result<String, String>),
}

type AdditionalModelLoadTask = Option<tokio::task::JoinHandle<AdditionalModelLoadResult>>;

type ServerStartResult = std::result::Result<
    (
        tokio::task::JoinHandle<athenas_core::Result<()>>,
        athenas_server::SharedModelManager,
        String,
        u16,
    ),
    athenas_core::AthenasError,
>;
type ServerStartTask = Option<tokio::task::JoinHandle<ServerStartResult>>;

#[derive(PartialEq)]
pub enum AppMode {
    Chat,
    ModelList,
    Browser,
    Server,
    Settings,
    Logs,
}

impl AppMode {
    pub fn tab_index(&self) -> usize {
        match self {
            AppMode::Chat => 0,
            AppMode::ModelList => 1,
            AppMode::Browser => 2,
            AppMode::Server => 3,
            AppMode::Settings => 4,
            AppMode::Logs => 5,
        }
    }
}

pub struct TuiApp {
    config: AppConfig,
    hardware: HardwareInfo,
    chat_state: ChatState,
    model_list_state: ModelListState,
    browser_state: ModelBrowserState,
    server_panel_state: ServerPanelState,
    settings_state: SettingsState,
    mode: AppMode,
    backend: Option<Box<dyn Backend>>,
    // Background download state
    download_progress_rx: Option<tokio::sync::mpsc::Receiver<athenas_hub::DownloadProgress>>,
    download_task: Option<
        tokio::task::JoinHandle<
            std::result::Result<std::path::PathBuf, athenas_core::AthenasError>,
        >,
    >,
    // Background server state
    server_handle: Option<tokio::task::JoinHandle<athenas_core::Result<()>>>,
    shared_model_manager: Option<athenas_server::SharedModelManager>,
    // External server state (detached process)
    server_state: Option<server_manager::ServerState>,
    server_health_task: Option<tokio::task::JoinHandle<Option<server_manager::ServerState>>>,
    // Background model loading state (chat mode)
    is_loading_model: bool,
    model_load_task: Option<
        tokio::task::JoinHandle<std::result::Result<Box<dyn Backend>, athenas_core::AthenasError>>,
    >,
    loading_spinner: usize,
    // Background additional model loading state (server panel)
    additional_model_load_task: AdditionalModelLoadTask,
    additional_model_name_hint: Option<String>,
    // Background server start task (server panel)
    server_start_task: ServerStartTask,
    // Logs page state
    logs_state: crate::log_buffer::LogsState,
    // Background server log tailer task
    server_log_tailer: Option<tokio::task::JoinHandle<()>>,
    // Background chat streaming state
    chat_stream_rx: Option<tokio::sync::mpsc::Receiver<StreamChunk>>,
    // API Key management modal
    api_key_modal: crate::api_key_modal::ApiKeyModalState,
}

impl TuiApp {
    pub fn new(config: AppConfig, hardware: HardwareInfo) -> Self {
        let registry = ModelRegistry::new(config.paths.models_dir.clone());
        let models = registry.list_local_models().unwrap_or_default();

        let mut model_list_state = ModelListState::default();
        model_list_state.set_models(models);

        let settings_state = SettingsState::new(config.clone()).with_hardware(hardware.clone());
        let server_panel_state = ServerPanelState::new(&config, hardware.clone());

        // Spawn a background health check to detect an already-running server
        let health_task = tokio::spawn(server_manager::check_running());

        // Create the log buffer (in-memory, circular — never persisted to disk)
        let log_buffer = crate::log_buffer::LogBuffer::new(2000);

        // Spawn a background task that tails the detached server's log file
        // and feeds new lines into the log buffer so they appear in the TUI
        // logs page alongside the TUI process's own tracing events.
        //
        // Uses spawn_blocking because the tailer does synchronous file I/O
        // (std::fs, BufReader) which would block the tokio async runtime
        // and freeze the TUI rendering if run in an async task.
        let tailer_buffer = log_buffer.clone();
        let log_tailer =
            tokio::task::spawn_blocking(move || Self::run_server_log_tailer(tailer_buffer));

        Self {
            config,
            hardware,
            chat_state: ChatState::default(),
            model_list_state,
            browser_state: ModelBrowserState::default(),
            server_panel_state,
            settings_state,
            mode: AppMode::Chat,
            backend: None,
            download_progress_rx: None,
            download_task: None,
            server_handle: None,
            shared_model_manager: None,
            server_state: None,
            server_health_task: Some(health_task),
            is_loading_model: false,
            model_load_task: None,
            loading_spinner: 0,
            additional_model_load_task: None,
            additional_model_name_hint: None,
            server_start_task: None,
            logs_state: crate::log_buffer::LogsState::new(log_buffer),
            server_log_tailer: Some(log_tailer),
            chat_stream_rx: None,
            api_key_modal: crate::api_key_modal::ApiKeyModalState::new(),
        }
    }

    pub fn with_log_buffer(
        config: AppConfig,
        hardware: HardwareInfo,
        log_buffer: crate::log_buffer::LogBuffer,
    ) -> Self {
        let mut app = Self::new(config, hardware);
        // Replace the default buffer with the one provided by the caller
        // (which already has a LogBufferLayer attached). The existing tailer
        // task is still running with the old buffer, so we spawn a new one
        // with the provided buffer and abort the old one.
        if let Some(old_tailer) = app.server_log_tailer.take() {
            old_tailer.abort();
        }
        let tailer_buffer = log_buffer.clone();
        app.server_log_tailer = Some(tokio::task::spawn_blocking(move || {
            Self::run_server_log_tailer(tailer_buffer)
        }));
        app.logs_state = crate::log_buffer::LogsState::new(log_buffer);
        app
    }

    /// Log a TUI event to the tracing system (appears in the logs page).
    fn log(&self, msg: &str) {
        tracing::info!("{}", msg);
    }

    /// Background thread that tails `~/.athenas/server.log` and pushes new
    /// lines into the log buffer. This makes the detached server's logs
    /// (including HTTP request logs) visible in the TUI logs page.
    ///
    /// This is a **blocking** function — it must run on a blocking thread
    /// (via `tokio::task::spawn_blocking`) so it doesn't freeze the async
    /// runtime and TUI rendering.
    ///
    /// The buffer is in-memory only — no logs are persisted by the TUI.
    ///
    /// Unlike the previous version, this does NOT block waiting for the log
    /// file to exist. If the file doesn't exist (no server running), it just
    /// polls periodically. This means the TUI's own tracing events still
    /// appear in the logs page even when no server is running.
    fn run_server_log_tailer(buffer: crate::log_buffer::LogBuffer) {
        let log_path = match dirs::home_dir() {
            Some(h) => h.join(".athenas").join("server.log"),
            None => return,
        };

        use std::io::{BufRead, BufReader, Seek, SeekFrom};

        // Track the file size we've already read. 0 means we haven't read
        // anything yet (or the file was truncated/recreated).
        let mut last_size: u64 = 0;
        // Track whether we've done the initial read of the file
        let mut initialized = false;

        loop {
            std::thread::sleep(std::time::Duration::from_millis(200));

            // Try to get file metadata. If the file doesn't exist, just
            // continue polling — don't block.
            let current_size = match std::fs::metadata(&log_path) {
                Ok(m) => m.len(),
                Err(_) => {
                    // File doesn't exist (no server running). Reset state
                    // so that when the file appears, we read from the start.
                    initialized = false;
                    last_size = 0;
                    continue;
                }
            };

            if !initialized {
                // First time the file exists — read from the beginning
                // so the user sees all available server logs immediately.
                initialized = true;
                last_size = 0;
            }

            // Check if the file was truncated/recreated (server restarted)
            if current_size < last_size {
                last_size = 0;
            }

            if current_size == last_size {
                continue;
            }

            // Read new content from last_size onwards
            if let Ok(mut f) = std::fs::File::open(&log_path) {
                let _ = f.seek(SeekFrom::Start(last_size));
                let mut reader = BufReader::new(f);
                let mut buf = String::new();
                loop {
                    buf.clear();
                    match reader.read_line(&mut buf) {
                        Ok(0) => break,
                        Ok(_) => {
                            let line = buf.trim_end();
                            if !line.is_empty() {
                                buffer.push_raw_line(line);
                            }
                        }
                        Err(_) => break,
                    }
                }
                last_size = current_size;
            }
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

        // Abort the background server log tailer
        if let Some(tailer) = self.server_log_tailer.take() {
            tailer.abort();
        }

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
            // Poll background download progress (non-blocking)
            self.poll_download_progress().await;

            // Poll server task status
            self.poll_server_status().await;

            // Poll server health check (detect external running server)
            self.poll_server_health().await;

            // Poll model loading task (chat mode)
            self.poll_model_loading().await;

            // Poll additional model loading task (server panel)
            self.poll_additional_model_loading().await;

            // Poll server start task (server panel)
            self.poll_server_start_task().await;

            // Poll chat stream
            self.poll_chat_stream().await;

            // Animate loading spinner
            if self.is_loading_model {
                self.loading_spinner = (self.loading_spinner + 1) % 4;
            }

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

                    // Tab navigation with F1-F5
                    if key.code == KeyCode::F(1) {
                        self.mode = AppMode::Chat;
                        continue;
                    }
                    if key.code == KeyCode::F(2) {
                        self.mode = AppMode::ModelList;
                        self.refresh_models();
                        continue;
                    }
                    if key.code == KeyCode::F(3) {
                        self.mode = AppMode::Browser;
                        continue;
                    }
                    if key.code == KeyCode::F(4) {
                        self.mode = AppMode::Server;
                        self.server_panel_state.refresh_models(&self.config);
                        continue;
                    }
                    if key.code == KeyCode::F(5) {
                        self.mode = AppMode::Settings;
                        continue;
                    }
                    if key.code == KeyCode::F(6) {
                        self.mode = AppMode::Logs;
                        continue;
                    }

                    // Global Tab cycling (skip when editing, in chat mode,
                    // or when the API key modal is open)
                    if key.code == KeyCode::Tab
                        && self.mode != AppMode::Chat
                        && !(self.mode == AppMode::Settings && self.settings_state.editing)
                        && !(self.mode == AppMode::Server && self.server_panel_state.editing)
                        && !self.api_key_modal.open
                    {
                        self.mode = match self.mode {
                            AppMode::Chat => AppMode::ModelList,
                            AppMode::ModelList => AppMode::Browser,
                            AppMode::Browser => AppMode::Server,
                            AppMode::Server => AppMode::Settings,
                            AppMode::Settings => AppMode::Logs,
                            AppMode::Logs => AppMode::Chat,
                        };
                        if matches!(self.mode, AppMode::ModelList) {
                            self.refresh_models();
                        }
                        if matches!(self.mode, AppMode::Server) {
                            self.server_panel_state.refresh_models(&self.config);
                        }
                        continue;
                    }

                    // API key modal intercepts all keys when open
                    if self.api_key_modal.open {
                        self.handle_api_key_modal_key(key).await;
                        continue;
                    }

                    match self.mode {
                        AppMode::Chat => self.handle_chat_key(key).await,
                        AppMode::ModelList => self.handle_model_list_key(key).await,
                        AppMode::Browser => self.handle_browser_key(key).await,
                        AppMode::Server => self.handle_server_key(key).await,
                        AppMode::Settings => self.handle_settings_key(key).await,
                        AppMode::Logs => self.handle_logs_key(key).await,
                    }
                }
            }
        }

        Ok(())
    }

    fn refresh_models(&mut self) {
        let registry = ModelRegistry::new(self.config.paths.models_dir.clone());
        let models = registry.list_local_models().unwrap_or_default();
        self.model_list_state.set_models(models);
    }

    fn render(&mut self, f: &mut ratatui::Frame) {
        let area = f.area();

        // Split off tab bar (1 line) + content
        let chunks = ratatui::layout::Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([
                ratatui::layout::Constraint::Length(1),
                ratatui::layout::Constraint::Min(3),
            ])
            .split(area);

        components::render_tab_bar(f, chunks[0], self.mode.tab_index());

        let content = chunks[1];
        match self.mode {
            AppMode::Chat => {
                // Update GPU info in chat state for status bar display
                if self.chat_state.gpu_info.is_empty() && !self.hardware.gpus.is_empty() {
                    self.chat_state.gpu_info = self
                        .hardware
                        .gpus
                        .iter()
                        .map(|g| format!("{} ({}MB)", g.name, g.vram_total_mb))
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                self.chat_state.gpu_runtime = self.config.inference.gpu_runtime.to_string();
                self.chat_state.gpu_layers = self.config.inference.default_gpu_layers;

                components::render_chat_area(
                    f,
                    content,
                    &mut self.chat_state,
                    self.is_loading_model,
                    self.loading_spinner,
                );
            }
            AppMode::ModelList => {
                components::render_model_list(
                    f,
                    content,
                    &self.model_list_state,
                    self.chat_state.current_model.as_deref(),
                    self.chat_state.current_backend.as_deref(),
                );
            }
            AppMode::Browser => {
                components::render_model_browser(f, content, &self.browser_state);
            }
            AppMode::Server => {
                components::render_server_panel(f, content, &self.server_panel_state);
            }
            AppMode::Settings => {
                components::render_settings(f, content, &self.settings_state);
            }
            AppMode::Logs => {
                components::render_logs(f, content, &self.logs_state);
            }
        }

        // Render API key modal on top of everything if open
        if self.api_key_modal.open {
            crate::api_key_modal::render_api_key_modal(f, &self.api_key_modal);
        }
    }

    async fn handle_chat_key(&mut self, key: event::KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                self.send_message().await;
            }
            // Tab toggles reasoning/thinking expansion on last assistant message
            KeyCode::Tab => {
                if let Some(msg) = self
                    .chat_state
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == "assistant" && !m.reasoning.is_empty())
                {
                    msg.reasoning_expanded = !msg.reasoning_expanded;
                }
            }
            // Up: scroll up one line (toward older messages)
            KeyCode::Up => {
                self.chat_state.auto_scroll = false;
                self.chat_state.scroll = self.chat_state.scroll.saturating_sub(1);
            }
            // Down: scroll down one line (toward newer messages)
            // If we reach the bottom, re-enable auto-scroll
            KeyCode::Down => {
                if self.chat_state.auto_scroll {
                    // Already following, nothing to do
                } else {
                    self.chat_state.scroll = self.chat_state.scroll.saturating_add(1);
                    // Render will clamp; if we hit bottom, auto-scroll re-enables
                    // We use a flag to detect this in render via a large scroll value
                }
            }
            KeyCode::Char(c)
                if !key.modifiers.contains(KeyModifiers::CONTROL) && !c.is_control() =>
            {
                self.chat_state.input_text.push(c);
                // Any typing re-enables auto-scroll
                self.chat_state.auto_scroll = true;
            }
            KeyCode::Backspace => {
                self.chat_state.input_text.pop();
            }
            KeyCode::Esc if self.chat_state.is_generating => {}
            _ => {}
        }
    }

    async fn handle_model_list_key(&mut self, key: event::KeyEvent) {
        match key.code {
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
                    self.log(&format!("Loading model from {}", path));
                    self.load_model(&path).await;
                    self.mode = AppMode::Chat;
                }
            }
            KeyCode::Delete | KeyCode::Char('d') => {
                if let Some(model) = self.model_list_state.selected() {
                    let name = model.name.clone();
                    let registry = ModelRegistry::new(self.config.paths.models_dir.clone());
                    match registry.remove_model(&name) {
                        Ok(_) => {
                            self.chat_state
                                .add_message("system", &format!("Model '{}' deleted.", name));
                            self.refresh_models();
                        }
                        Err(e) => {
                            self.chat_state.add_message(
                                "system",
                                &format!("Failed to delete model '{}': {}", name, e),
                            );
                        }
                    }
                }
            }
            KeyCode::Char('u') => {
                if self.backend.is_some() {
                    let name = self.chat_state.current_model.clone().unwrap_or_default();
                    let backend_name = self.chat_state.current_backend.clone().unwrap_or_default();
                    // Drop the backend — this frees the model from memory
                    self.backend = None;
                    self.chat_state.current_model = None;
                    self.chat_state.current_backend = None;
                    // Stop any ongoing stream
                    self.chat_stream_rx = None;
                    self.log(&format!(
                        "Model '{}' [{}] unloaded from memory",
                        name, backend_name
                    ));
                    self.chat_state.add_message(
                        "system",
                        &format!("Model '{}' [{}] unloaded from memory.", name, backend_name),
                    );
                } else {
                    self.chat_state
                        .add_message("system", "No model is currently loaded.");
                }
            }
            KeyCode::Esc => {
                self.mode = AppMode::Chat;
            }
            _ => {}
        }
    }

    async fn handle_settings_key(&mut self, key: event::KeyEvent) {
        if self.settings_state.editing {
            match key.code {
                KeyCode::Esc => {
                    self.settings_state.cancel_edit();
                }
                KeyCode::Enter => {
                    if let Err(e) = self.settings_state.save_edit() {
                        self.settings_state.status_message = Some(e);
                    } else {
                        // Sync settings_state.config back to self.config
                        // so that load_model uses the updated values
                        self.config = self.settings_state.config.clone();
                    }
                }
                KeyCode::Backspace => {
                    self.settings_state.edit_buffer.pop();
                }
                KeyCode::Char(c) => {
                    if self.settings_state.edit_buffer == "[hidden — type to replace]" {
                        self.settings_state.edit_buffer.clear();
                    }
                    self.settings_state.edit_buffer.push(c);
                }
                _ => {}
            }
        } else {
            match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.settings_state.next();
                    self.settings_state.status_message = None;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.settings_state.previous();
                    self.settings_state.status_message = None;
                }
                KeyCode::Enter => {
                    // Device and GpuRuntime are toggles — cycle values
                    // directly on Enter without entering edit mode.
                    let field = self.settings_state.current_field().clone();
                    if field == crate::settings::SettingsField::Device {
                        let current = self.settings_state.config.inference.default_gpu_layers;
                        let new_val = if current == 0 { -1 } else { 0 };
                        self.settings_state.config.inference.default_gpu_layers = new_val;
                        let device = if new_val == 0 { "CPU" } else { "GPU" };
                        if let Err(e) = self.settings_state.config.save() {
                            self.settings_state.status_message =
                                Some(format!("Failed to save: {}", e));
                        } else {
                            self.settings_state.status_message =
                                Some(format!("Device set to {} — saved", device));
                            self.log(&format!("Device changed to {}", device));
                            // Sync back to self.config so load_model uses it
                            self.config = self.settings_state.config.clone();
                        }
                    } else if field == crate::settings::SettingsField::GpuRuntime {
                        use athenas_core::GpuRuntime;
                        let next = match self.settings_state.config.inference.gpu_runtime {
                            GpuRuntime::Auto => GpuRuntime::Cuda,
                            GpuRuntime::Cuda => GpuRuntime::Rocm,
                            GpuRuntime::Rocm => GpuRuntime::Vulkan,
                            GpuRuntime::Vulkan => GpuRuntime::Metal,
                            GpuRuntime::Metal => GpuRuntime::Cpu,
                            GpuRuntime::Cpu => GpuRuntime::Auto,
                        };
                        self.settings_state.config.inference.gpu_runtime = next;
                        let rt_str = next.to_string();
                        if let Err(e) = self.settings_state.config.save() {
                            self.settings_state.status_message =
                                Some(format!("Failed to save: {}", e));
                        } else {
                            self.settings_state.status_message =
                                Some(format!("GPU runtime set to {} — saved", rt_str));
                            self.log(&format!("GPU runtime changed to {}", rt_str));
                            // Sync back to self.config so load_model uses it
                            self.config = self.settings_state.config.clone();
                        }
                    } else {
                        self.settings_state.start_edit();
                    }
                }
                KeyCode::Esc => {
                    self.mode = AppMode::Chat;
                }
                _ => {}
            }
        }
    }

    async fn handle_logs_key(&mut self, key: event::KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.logs_state.scroll_up(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.logs_state.scroll_down(1);
            }
            KeyCode::PageUp => {
                self.logs_state.scroll_up(10);
            }
            KeyCode::PageDown => {
                self.logs_state.scroll_down(10);
            }
            KeyCode::End => {
                self.logs_state.scroll_to_bottom();
            }
            KeyCode::Char('c') => {
                self.logs_state.clear();
            }
            KeyCode::Char('a') => {
                if self.logs_state.auto_scroll {
                    self.logs_state.scroll_up(1);
                } else {
                    self.logs_state.scroll_to_bottom();
                }
            }
            KeyCode::Esc => {
                self.mode = AppMode::Chat;
            }
            _ => {}
        }
    }

    async fn handle_browser_key(&mut self, key: event::KeyEvent) {
        match &self.browser_state.phase {
            BrowserPhase::Search => match key.code {
                KeyCode::Enter => {
                    let query = self.browser_state.search_input.trim().to_string();
                    if !query.is_empty() {
                        self.browser_state.status_message = Some("Searching...".to_string());
                        self.browser_state.status_is_error = false;
                        self.perform_search(&query).await;
                    }
                }
                KeyCode::Backspace => {
                    self.browser_state.search_input.pop();
                }
                KeyCode::Char('g') | KeyCode::Char('G') => {
                    self.browser_state.gguf_only = !self.browser_state.gguf_only;
                }
                KeyCode::Esc => {
                    self.mode = AppMode::Chat;
                }
                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.browser_state.search_input.clear();
                }
                KeyCode::Char(c) => {
                    self.browser_state.search_input.push(c);
                }
                _ => {}
            },
            BrowserPhase::Results => match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.browser_state.next_result();
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.browser_state.prev_result();
                }
                KeyCode::Enter => {
                    if let Some(result) = self.browser_state.selected_result() {
                        let repo_id = result.id.clone();
                        self.browser_state.status_message = Some("Loading files...".to_string());
                        self.browser_state.status_is_error = false;
                        self.list_files(&repo_id).await;
                    }
                }
                KeyCode::Esc => {
                    self.browser_state.back_to_search_edit();
                }
                KeyCode::Char('/') => {
                    self.browser_state.back_to_search_edit();
                }
                KeyCode::Char('r') | KeyCode::Char('R') => {
                    self.browser_state.reset_search();
                }
                _ => {}
            },
            BrowserPhase::SelectFile => match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.browser_state.next_file();
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.browser_state.prev_file();
                }
                KeyCode::Enter => {
                    if let Some((filename, _)) = self
                        .browser_state
                        .file_options
                        .get(self.browser_state.file_selected)
                        .cloned()
                    {
                        let repo_id = self
                            .browser_state
                            .selected_result()
                            .map(|r| r.id.clone())
                            .unwrap_or_default();
                        self.browser_state.phase = BrowserPhase::Downloading;
                        self.browser_state.download_filename = Some(filename.clone());
                        self.browser_state.download_progress = None;
                        self.browser_state.status_message = None;
                        self.browser_state.status_is_error = false;
                        self.start_download(&repo_id, &filename);
                    }
                }
                KeyCode::Esc => {
                    self.browser_state.phase = BrowserPhase::Results;
                }
                _ => {}
            },
            BrowserPhase::Downloading => {
                if key.code == KeyCode::Esc {
                    // Abort: drop the receiver and task
                    self.download_progress_rx = None;
                    if let Some(handle) = self.download_task.take() {
                        handle.abort();
                    }
                    self.browser_state.phase = BrowserPhase::Results;
                    self.browser_state.download_progress = None;
                    self.browser_state.download_filename = None;
                    self.browser_state.status_message = Some("Download cancelled".to_string());
                    self.browser_state.status_is_error = false;
                }
            }
        }
    }

    async fn perform_search(&mut self, query: &str) {
        let token = self.config.huggingface.token.clone();
        let client = athenas_hub::HuggingFaceClient::new(token);
        let filters = athenas_hub::ModelSearchFilters {
            pipeline_tag: None,
            library_name: None,
            gguf_only: self.browser_state.gguf_only,
            safetensors_only: false,
        };

        match client.search_models(query, &filters).await {
            Ok(results) => {
                self.browser_state.search_results = results;
                self.browser_state.results_selected = 0;
                self.browser_state.phase = BrowserPhase::Results;
                self.browser_state.status_message = None;
                self.browser_state.status_is_error = false;
            }
            Err(e) => {
                self.browser_state.status_message = Some(format!("Search failed: {}", e));
                self.browser_state.status_is_error = true;
            }
        }
    }

    async fn list_files(&mut self, repo_id: &str) {
        let token = self.config.huggingface.token.clone();
        let client = athenas_hub::HuggingFaceClient::new(token);

        match client.get_model_files(repo_id, "main").await {
            Ok(files) => {
                let gguf_files: Vec<(String, Option<u64>)> = files
                    .iter()
                    .filter(|f| f.path.ends_with(".gguf"))
                    .map(|f| {
                        (
                            f.path.clone(),
                            f.size.or(f.lfs.as_ref().and_then(|l| l.size)),
                        )
                    })
                    .collect();

                if gguf_files.is_empty() {
                    self.browser_state.status_message = Some(
                        "No GGUF files found in this repo. llama-server requires GGUF format. \
                         Try searching for GGUF-quantized versions."
                            .to_string(),
                    );
                    self.browser_state.status_is_error = true;
                } else {
                    self.browser_state.file_options = gguf_files;
                    self.browser_state.file_selected = 0;
                    self.browser_state.phase = BrowserPhase::SelectFile;
                    self.browser_state.status_message = None;
                    self.browser_state.status_is_error = false;
                }
            }
            Err(e) => {
                self.browser_state.status_message = Some(format!("Failed to list files: {}", e));
                self.browser_state.status_is_error = true;
            }
        }
    }

    fn start_download(&mut self, repo_id: &str, filename: &str) {
        let token = self.config.huggingface.token.clone();
        let client = athenas_hub::HuggingFaceClient::new(token);
        let downloader =
            athenas_hub::ModelDownloader::new(client.clone(), self.config.paths.models_dir.clone());

        let (tx, rx) = tokio::sync::mpsc::channel::<athenas_hub::DownloadProgress>(10);

        let repo_id_owned = repo_id.to_string();
        let filename_owned = filename.to_string();

        // Also fetch mmproj files for multimodal support
        let client_for_mmproj = client.clone();
        let downloader_clone = athenas_hub::ModelDownloader::new(
            client_for_mmproj.clone(),
            self.config.paths.models_dir.clone(),
        );
        let tx_clone = tx.clone();

        let download_task = tokio::spawn(async move {
            let result = downloader
                .download_model(&repo_id_owned, &filename_owned, "main", Some(tx))
                .await;

            if result.is_ok() {
                // Auto-download mmproj files if present
                if let Ok(files) = client_for_mmproj
                    .get_model_files(&repo_id_owned, "main")
                    .await
                {
                    let mmproj_files: Vec<_> = files
                        .iter()
                        .filter(|f| {
                            f.r#type == "file"
                                && f.path.to_lowercase().contains("mmproj")
                                && (f.path.ends_with(".gguf") || f.path.ends_with(".bin"))
                        })
                        .collect();

                    for mmproj in &mmproj_files {
                        let _ = downloader_clone
                            .download_model(
                                &repo_id_owned,
                                &mmproj.path,
                                "main",
                                Some(tx_clone.clone()),
                            )
                            .await;
                    }
                }
            }

            result
        });

        self.download_progress_rx = Some(rx);
        self.download_task = Some(download_task);
    }

    async fn poll_download_progress(&mut self) {
        if self.download_progress_rx.is_none() {
            return;
        }

        // Drain all pending progress updates (non-blocking)
        while let Ok(progress) = self.download_progress_rx.as_mut().unwrap().try_recv() {
            self.browser_state.download_progress = Some((
                progress.downloaded_bytes,
                progress.total_bytes.unwrap_or(0),
                progress.speed_mbps,
            ));
        }

        // Check if download task is done
        if let Some(handle) = &mut self.download_task {
            if handle.is_finished() {
                let result = handle.await;
                self.download_task = None;
                self.download_progress_rx = None;

                match result {
                    Ok(Ok(path)) => {
                        self.browser_state.phase = BrowserPhase::Results;
                        self.browser_state.download_progress = None;
                        self.browser_state.download_filename = None;
                        self.browser_state.status_message =
                            Some(format!("Downloaded to: {}", path.display()));
                        self.browser_state.status_is_error = false;
                        self.refresh_models();
                        self.server_panel_state.refresh_models(&self.config);
                    }
                    Ok(Err(e)) => {
                        self.browser_state.phase = BrowserPhase::Results;
                        self.browser_state.download_progress = None;
                        self.browser_state.download_filename = None;
                        self.browser_state.status_message = Some(format!("Download failed: {}", e));
                        self.browser_state.status_is_error = true;
                    }
                    Err(e) => {
                        self.browser_state.phase = BrowserPhase::Results;
                        self.browser_state.download_progress = None;
                        self.browser_state.download_filename = None;
                        self.browser_state.status_message = Some(format!("Task failed: {}", e));
                        self.browser_state.status_is_error = true;
                    }
                }
            }
        }
    }

    async fn send_message(&mut self) {
        let text = self.chat_state.input_text.trim().to_string();
        if text.is_empty() {
            return;
        }

        // Re-enable auto-scroll for new messages
        self.chat_state.auto_scroll = true;
        self.chat_state.scroll = 0;

        // Handle commands
        if text.starts_with('/') {
            self.handle_command(&text).await;
            return;
        }

        if self.backend.is_none() {
            // Check if the server has models loaded — use those instead
            if let Some(mgr) = &self.shared_model_manager {
                let m = mgr.lock().await;
                if m.has_models() {
                    // Only show the message once when current_model is not set
                    if self.chat_state.current_model.is_none() {
                        let default_name = m
                            .default_id()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        self.chat_state.add_message(
                            "system",
                            &format!(
                                "Using server model '{}' for inference. \
                                 The server is running with {} model(s) loaded.",
                                default_name,
                                m.count()
                            ),
                        );
                    }
                } else {
                    self.chat_state.add_message(
                        "system",
                        "No model loaded. Press F2 to select a model, or start the server (F4) to load one.",
                    );
                    return;
                }
            } else if let Some(ref state) = self.server_state {
                // Detached server is running — use HTTP API
                // Only show the message once when current_model is not set
                if self.chat_state.current_model.is_none() {
                    self.chat_state.add_message(
                        "system",
                        &format!(
                            "Using remote server model '{}' ({}:{})",
                            state.model, state.host, state.port
                        ),
                    );
                    self.chat_state.current_model = Some(state.model.clone());
                    self.chat_state.current_backend = Some("remote".to_string());
                }
            } else {
                self.chat_state
                    .add_message("system", "No model loaded. Press F2 to select a model.");
                return;
            }
        }

        if self.chat_state.is_generating {
            return;
        }

        self.chat_state.add_message("user", &text);
        self.chat_state.input_text.clear();
        self.chat_state.is_generating = true;
        self.chat_state.generation_start = Some(std::time::Instant::now());

        // Build chat request from current messages
        // Filter out ALL system messages — they are TUI informational messages
        // (welcome, model loaded, errors) not meant for the model's context.
        // Many model chat templates (e.g. Qwen) require system messages only
        // at the beginning and reject them if placed after user/assistant turns.
        let messages: Vec<ChatMessage> = self
            .chat_state
            .messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| {
                let role = match m.role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                };
                ChatMessage {
                    role,
                    content: MessageContent::Text(m.content.clone()),
                }
            })
            .collect();

        let req = ChatRequest {
            model: String::new(),
            messages,
            temperature: Some(self.config.inference.default_temperature),
            top_p: Some(self.config.inference.default_top_p),
            max_tokens: Some(self.config.inference.default_max_tokens),
            stream: self.config.inference.streaming_enabled,
            stop: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            grammar: None,
        };

        // Get a backend reference: prefer local backend, then in-process server, then detached server
        let backend_ref: Option<Box<dyn Backend>> = if let Some(ref b) = self.backend {
            Some(b.boxed_clone())
        } else if let Some(mgr) = &self.shared_model_manager {
            let m = mgr.lock().await;
            m.get(None).map(|b| b.boxed_clone())
        } else if let Some(ref state) = self.server_state {
            let api_key = if self.server_panel_state.api_key.is_empty() {
                None
            } else {
                Some(self.server_panel_state.api_key.as_str())
            };
            Some(Box::new(RemoteBackend::new(
                &state.host,
                state.port,
                &state.model,
                api_key,
            )))
        } else {
            None
        };

        let Some(backend) = backend_ref else {
            self.chat_state
                .add_message("system", "No backend available for inference.");
            return;
        };

        if !self.config.inference.streaming_enabled {
            // Non-streaming: spawn chat() in background, show result when done
            let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
            tokio::spawn(async move {
                match backend.chat(req).await {
                    Ok(resp) => {
                        let _ = tx
                            .send(StreamChunk {
                                text: resp.message.content.as_text(),
                                done: false,
                                is_reasoning: false,
                                stats: None,
                            })
                            .await;
                        let _ = tx
                            .send(StreamChunk {
                                text: String::new(),
                                done: true,
                                is_reasoning: false,
                                stats: Some(resp.stats),
                            })
                            .await;
                    }
                    Err(e) => {
                        tracing::error!("Chat error: {}", e);
                    }
                }
            });
            self.chat_stream_rx = Some(rx);
            return;
        }

        // Start streaming in background — store receiver for polling
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        tokio::spawn(async move {
            if let Err(e) = backend.chat_stream(req, tx).await {
                tracing::error!("Chat stream error: {}", e);
            }
        });

        self.chat_stream_rx = Some(rx);
    }

    async fn poll_chat_stream(&mut self) {
        if !self.chat_state.is_generating {
            return;
        }

        // Timeout check: only abort if NO tokens received within 120s.
        // If the model is actively generating (reasoning or content), keep waiting.
        if let Some(start) = self.chat_state.generation_start {
            let elapsed = start.elapsed().as_secs();
            let has_output = !self.chat_state.streaming_text.is_empty()
                || !self.chat_state.streaming_reasoning.is_empty();
            if elapsed > 120 && !has_output {
                self.chat_state.add_message(
                    "system",
                    "Request timed out (120s with no output). The model may not be responding. \
                     Try a smaller model, reduce context size, or disable reasoning in Settings.",
                );
                self.chat_state.finalize_streaming();
                self.chat_stream_rx = None;
                return;
            }
        }

        if let Some(rx) = &mut self.chat_stream_rx {
            // Non-blocking: try to receive available chunks without waiting
            while let Ok(chunk) = rx.try_recv() {
                if chunk.done {
                    if let Some(stats) = chunk.stats {
                        self.chat_state.tokens_per_second = Some(stats.tokens_per_second);
                    }
                    self.chat_state.finalize_streaming();
                    self.chat_stream_rx = None;
                    return;
                } else {
                    if chunk.is_reasoning {
                        self.chat_state.append_reasoning(&chunk.text);
                    } else {
                        self.chat_state.append_streaming(&chunk.text);
                    }
                    // Update tok/s live during streaming
                    if let Some(stats) = &chunk.stats {
                        self.chat_state.tokens_per_second = Some(stats.tokens_per_second);
                    }
                }
            }

            // Check if the sender was dropped (stream ended without done chunk)
            if rx.is_closed() {
                if self.chat_state.is_generating {
                    self.chat_state.finalize_streaming();
                }
                self.chat_stream_rx = None;
            }
        }
    }

    async fn handle_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        match parts[0] {
            "/clear" => {
                self.chat_state.clear();
            }
            "/unload" => {
                if let Some(mut backend) = self.backend.take() {
                    let model_name = backend
                        .model_info()
                        .map(|i| i.name.clone())
                        .unwrap_or_default();
                    match backend.unload_model().await {
                        Ok(()) => {
                            self.chat_state.current_model = None;
                            self.chat_state.current_backend = None;
                            self.chat_state.add_message(
                                "system",
                                &format!("Model '{}' unloaded from memory.", model_name),
                            );
                        }
                        Err(e) => {
                            self.chat_state
                                .add_message("system", &format!("Failed to unload model: {}", e));
                            self.backend = Some(backend);
                        }
                    }
                } else {
                    self.chat_state
                        .add_message("system", "No model is currently loaded.");
                }
            }
            "/model" | "/models" => {
                self.mode = AppMode::ModelList;
                self.refresh_models();
            }
            "/browser" => {
                self.mode = AppMode::Browser;
            }
            "/server" => {
                self.mode = AppMode::Server;
                self.server_panel_state.refresh_models(&self.config);
            }
            "/settings" => {
                self.mode = AppMode::Settings;
            }
            "/logs" => {
                self.mode = AppMode::Logs;
            }
            "/help" => {
                self.chat_state.add_message(
                    "system",
                    "Commands: /clear, /unload, /model, /models, /browser, /server, /settings, /logs, /help, /quit\n\
                     F1: Chat | F2: Models | F3: Browser | F4: Server | F5: Settings | F6: Logs | Ctrl+C: Quit",
                );
            }
            "/quit" => {
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
        if self.is_loading_model {
            self.chat_state
                .add_message("system", "Already loading a model, please wait...");
            return;
        }

        // === Resource protections ===

        // Skip auto-capping if user disabled it
        let auto_limits = self.config.inference.auto_resource_limits;

        // 1. Check model file size vs available RAM
        let model_size_mb = std::fs::metadata(path)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);

        let avail_mb = self.hardware.memory_available_mb;
        let total_mb = self.hardware.memory_total_mb;

        // Model needs roughly 1.5x its file size in RAM (weights + KV cache + overhead)
        // For Q4 models: file size ≈ weights, context adds ~ctx_size * 2KB * layers
        let estimated_needed_mb =
            model_size_mb + ((self.config.inference.default_context_size as u64 / 1024) * 64);

        if auto_limits && avail_mb > 0 && estimated_needed_mb > avail_mb {
            self.chat_state.add_message(
                "system",
                &format!(
                    "⚠ Not enough RAM to load this model safely.\n\
                     Model: {}MB, estimated need: {}MB, available: {}MB\n\
                     Try a smaller model, smaller context size, or close other applications.",
                    model_size_mb, estimated_needed_mb, avail_mb
                ),
            );
            return;
        }

        // 2. Cap threads based on cpu_reserve_cores
        let mut threads = self.config.inference.default_threads;
        if auto_limits {
            let max_threads = self
                .hardware
                .cpus
                .saturating_sub(self.config.inference.cpu_reserve_cores)
                .max(1);
            if threads > max_threads {
                threads = max_threads;
            }
        }

        // 3. Cap context size based on available memory
        let mut context_size = self.config.inference.default_context_size;
        if auto_limits {
            let max_ctx_by_mem = if total_mb > 0 {
                // Reserve model size + ram_reserve_mb, allow up to 50% of remaining for context
                let reserved = model_size_mb + self.config.inference.ram_reserve_mb;
                let usable = total_mb.saturating_sub(reserved);
                // Rough: ctx_mb = usable * 0.4, ctx = ctx_mb / 64 * 1024
                ((usable * 1024) / (64 * 1024 / 1024)) as u32 * 1024
            } else {
                8192
            };
            if context_size > max_ctx_by_mem && max_ctx_by_mem > 0 {
                context_size = max_ctx_by_mem.max(512);
            }
        }

        // 4. Cap batch size — large batches consume more memory
        let mut batch_size = self.config.inference.default_batch_size;
        if batch_size > context_size {
            batch_size = context_size;
        }

        self.chat_state.add_message(
            "system",
            &format!(
                "Loading model: {}...\n\
                 Resource limits: {} threads, {} ctx, {} batch (RAM: {}MB/{}MB)",
                path, threads, context_size, batch_size, avail_mb, total_mb
            ),
        );
        self.is_loading_model = true;
        self.loading_spinner = 0;

        let backend_type = self.config.inference.default_backend;
        let hardware = self.hardware.clone();
        let load_config = ModelLoadConfig {
            model_path: path.to_string(),
            gpu_layers: self.config.inference.default_gpu_layers,
            gpu_runtime: self.config.inference.gpu_runtime,
            gpu_device: self.config.inference.gpu_device,
            context_size,
            batch_size,
            threads,
            flash_attention: self.config.inference.flash_attention,
            use_mmap: true,
            use_mlock: false,
            reasoning_enabled: self.config.inference.reasoning_enabled,
            reasoning_budget: self.config.inference.reasoning_budget,
            mmproj_path: None,
            lora_paths: Vec::new(),
            parallel_slots: 1,
        };

        let task = tokio::spawn(async move {
            let mut backend = BackendFactory::create(backend_type, &hardware)?;
            backend.load_model(load_config).await?;
            Ok::<Box<dyn Backend>, athenas_core::AthenasError>(backend)
        });

        self.model_load_task = Some(task);
    }

    async fn poll_model_loading(&mut self) {
        if !self.is_loading_model {
            return;
        }

        if let Some(task) = &mut self.model_load_task {
            if !task.is_finished() {
                return;
            }

            // Task is done, take it and get the result
            let task = self.model_load_task.take().unwrap();
            match task.await {
                Ok(Ok(backend)) => {
                    let info = backend.model_info();
                    let gpu_layers = info.as_ref().map(|i| i.gpu_layers).unwrap_or(-1);
                    if let Some(ref i) = info {
                        self.chat_state.current_model = Some(i.name.clone());
                        self.chat_state.current_backend = Some(i.backend_name.clone());
                        self.log(&format!(
                            "Model '{}' loaded successfully [{}]",
                            i.name, i.backend_name
                        ));
                    }
                    // Show GPU/device info in the system message
                    let gpu_msg = if self.hardware.gpus.is_empty() {
                        "CPU-only mode (no GPU detected)".to_string()
                    } else {
                        let gpu_name = &self.hardware.gpus[0].name;
                        let layers_str = if gpu_layers < 0 {
                            "all layers".to_string()
                        } else if gpu_layers == 0 {
                            "0 layers (CPU)".to_string()
                        } else {
                            format!("{} layers", gpu_layers)
                        };
                        format!(
                            "GPU: {} | Runtime: {} | {} on GPU",
                            gpu_name, self.config.inference.gpu_runtime, layers_str
                        )
                    };
                    self.chat_state.add_message(
                        "system",
                        &format!("Model loaded successfully!\n  {}", gpu_msg),
                    );
                    self.backend = Some(backend);
                    self.is_loading_model = false;
                }
                Ok(Err(e)) => {
                    self.log(&format!("Failed to load model: {}", e));
                    self.chat_state
                        .add_message("system", &format!("Failed to load model: {}", e));
                    self.is_loading_model = false;
                }
                Err(e) => {
                    self.log(&format!("Model loading task crashed: {}", e));
                    self.chat_state
                        .add_message("system", &format!("Model loading task crashed: {}", e));
                    self.is_loading_model = false;
                }
            }
        }
    }

    async fn handle_server_key(&mut self, key: event::KeyEvent) {
        if self.server_panel_state.editing {
            match key.code {
                KeyCode::Esc => {
                    self.server_panel_state.cancel_edit();
                }
                KeyCode::Enter => {
                    if let Err(e) = self.server_panel_state.save_edit() {
                        self.server_panel_state.status_message = Some(e);
                    }
                }
                KeyCode::Backspace => {
                    self.server_panel_state.edit_buffer.pop();
                }
                KeyCode::Char(c) => {
                    if self.server_panel_state.edit_buffer == "[type to replace]" {
                        self.server_panel_state.edit_buffer.clear();
                    }
                    self.server_panel_state.edit_buffer.push(c);
                }
                _ => {}
            }
        } else {
            let field = self.server_panel_state.current_field().clone();
            match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.server_panel_state.next();
                    self.server_panel_state.status_message = None;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.server_panel_state.previous();
                    self.server_panel_state.status_message = None;
                }
                KeyCode::Left | KeyCode::Char('h') => match field {
                    ConfigField::ModelSelection => {
                        self.server_panel_state.select_model_prev();
                    }
                    ConfigField::UnloadModel => {
                        self.server_panel_state.unload_select_prev();
                    }
                    ConfigField::SetDefaultModel => {
                        self.server_panel_state.default_select_prev();
                    }
                    _ => {}
                },
                KeyCode::Right | KeyCode::Char('l') => match field {
                    ConfigField::ModelSelection => {
                        self.server_panel_state.select_model_next();
                    }
                    ConfigField::UnloadModel => {
                        self.server_panel_state.unload_select_next();
                    }
                    ConfigField::SetDefaultModel => {
                        self.server_panel_state.default_select_next();
                    }
                    _ => {}
                },
                KeyCode::Enter => {
                    if field.is_toggle() {
                        self.server_panel_state.toggle();
                    } else if field == ConfigField::GpuRuntime {
                        // Cycle through GPU runtimes on Enter
                        use athenas_core::GpuRuntime;
                        let next = match self.server_panel_state.gpu_runtime {
                            GpuRuntime::Auto => GpuRuntime::Cuda,
                            GpuRuntime::Cuda => GpuRuntime::Rocm,
                            GpuRuntime::Rocm => GpuRuntime::Vulkan,
                            GpuRuntime::Vulkan => GpuRuntime::Metal,
                            GpuRuntime::Metal => GpuRuntime::Cpu,
                            GpuRuntime::Cpu => GpuRuntime::Auto,
                        };
                        self.server_panel_state.gpu_runtime = next;
                        self.server_panel_state.status_message =
                            Some(format!("GPU runtime: {} (saved on server start)", next));
                    } else if field.is_editable() {
                        self.server_panel_state.start_edit();
                    } else if field == ConfigField::StartServer {
                        self.start_server().await;
                    } else if field == ConfigField::StopServer {
                        self.stop_server();
                    } else if field == ConfigField::LoadAdditionalModel {
                        self.load_additional_model().await;
                    } else if field == ConfigField::UnloadModel {
                        self.unload_model_action().await;
                    } else if field == ConfigField::SetDefaultModel {
                        self.set_default_model_action().await;
                    } else if field == ConfigField::ManageApiKeys {
                        self.api_key_modal.open();
                        self.refresh_api_keys().await;
                    }
                }
                KeyCode::Esc => {
                    self.mode = AppMode::Chat;
                }
                _ => {}
            }
        }
    }

    async fn start_server(&mut self) {
        if self.server_panel_state.phase == ServerPhase::Running {
            self.server_panel_state.status_message = Some("Server is already running".to_string());
            return;
        }

        // Don't start if already loading
        if self.server_start_task.is_some() {
            self.server_panel_state.status_message =
                Some("Server is already starting, please wait...".to_string());
            return;
        }

        let model_path = match self.server_panel_state.selected_model_path() {
            Some(p) => p,
            None => {
                self.server_panel_state.status_message =
                    Some("No model selected. Use Left/Right to pick a model.".to_string());
                return;
            }
        };

        // Get model name for display
        let model_name = self
            .server_panel_state
            .models
            .get(self.server_panel_state.model_selected)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| model_path.clone());

        self.server_panel_state.phase = ServerPhase::LoadingModel;
        self.server_panel_state.status_message =
            Some(format!("Starting server with model: {}...", model_name));

        // Save server config so it persists across restarts
        let server_config = self.server_panel_state.build_app_config(&self.config);
        if let Err(e) = server_config.save() {
            tracing::warn!("Failed to save server config: {}", e);
        }
        self.config = server_config;

        let host = self.server_panel_state.host.clone();
        let port = self.server_panel_state.port;
        let backend_str = match self.server_panel_state.backend {
            athenas_core::BackendType::Auto => "auto",
            athenas_core::BackendType::LlamaCpp => "llama.cpp",
            athenas_core::BackendType::Vllm => "vllm",
        };
        let gpu_layers = self.server_panel_state.gpu_layers;
        let gpu_runtime = self.server_panel_state.gpu_runtime.to_string();
        let gpu_device = self.server_panel_state.gpu_device;
        let context_size = self.server_panel_state.context_size;
        let max_concurrent = Some(self.server_panel_state.max_concurrent);
        let rate_limit = Some(self.server_panel_state.rate_limit);
        let timeout_secs = Some(self.server_panel_state.timeout_secs);
        let max_body_size_mb = Some(self.server_panel_state.max_body_size);

        // Start the server as a detached child process
        match server_manager::start_detached(
            &model_path,
            &host,
            port,
            backend_str,
            gpu_layers,
            &gpu_runtime,
            gpu_device,
            context_size,
            max_concurrent,
            rate_limit,
            timeout_secs,
            max_body_size_mb,
        ) {
            Ok(state) => {
                self.server_panel_state.server_url = Some(format!("http://{}:{}", host, port));
                // Keep LoadingModel phase — the detached process is still loading the model.
                // We'll poll the health endpoint until it responds, then switch to Running.
                let health_pid = state.pid;
                self.server_panel_state.status_message =
                    Some(format!("Loading model (PID: {})...", health_pid));
                self.server_panel_state.loaded_model_name = Some(model_name.clone());
                self.server_panel_state.loaded_backend_name = Some(backend_str.to_string());
                self.server_state = Some(state);

                // Spawn a background task to poll the health endpoint
                let health_host = host.clone();
                let health_port = port;
                let health_state = self.server_state.clone();
                let health_task = tokio::spawn(async move {
                    let url = format!("http://{}:{}/v1/health", health_host, health_port);
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(3))
                        .build()
                        .ok()?;
                    // Poll for up to 5 minutes (300 attempts * 1s interval)
                    for _ in 0..300 {
                        // Check if the server process is still alive
                        if !server_manager::is_process_alive(health_pid) {
                            tracing::error!(
                                "Server process (PID: {}) died during startup. \
                                 Check ~/.athenas/server.log for details.",
                                health_pid
                            );
                            return None;
                        }
                        if client.get(&url).send().await.is_ok() {
                            return health_state;
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                    None
                });
                self.server_health_task = Some(health_task);

                // Update chat state to show server is available
                self.chat_state.current_model = Some(model_name);
                self.chat_state.current_backend = Some(backend_str.to_string());
            }
            Err(e) => {
                self.server_panel_state.phase = ServerPhase::Error;
                self.server_panel_state.status_message =
                    Some(format!("Failed to start server: {}", e));
            }
        }
    }

    async fn poll_server_health(&mut self) {
        let task = match self.server_health_task.take() {
            Some(t) => t,
            None => return,
        };
        if !task.is_finished() {
            self.server_health_task = Some(task);
            return;
        }

        match task.await {
            Ok(Some(state)) => {
                // A running server was detected
                self.server_state = Some(state.clone());
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state.server_url =
                    Some(format!("http://{}:{}", state.host, state.port));
                self.server_panel_state.loaded_model_name = Some(state.model.clone());
                self.server_panel_state.loaded_backend_name = Some(state.backend.clone());
                self.server_panel_state.status_message =
                    Some(format!("Server running (PID: {})", state.pid));
                self.chat_state.current_model = Some(state.model);
                self.chat_state.current_backend = Some(state.backend);
                // Fetch loaded models list from the detached server
                self.refresh_remote_loaded_models().await;
            }
            Ok(None) => {
                // No running server found
                if self.server_panel_state.phase == ServerPhase::LoadingModel {
                    // We were waiting for a server we started — it timed out
                    self.server_panel_state.phase = ServerPhase::Error;
                    self.server_panel_state.status_message = Some(
                        "Server failed to start. Check ~/.athenas/server.log for details."
                            .to_string(),
                    );
                    self.server_state = None;
                }
                // Otherwise it was just a startup check — no server running, that's fine
            }
            Err(e) => {
                tracing::warn!("Server health check task failed: {}", e);
            }
        }
    }

    async fn poll_server_start_task(&mut self) {
        if self.server_start_task.is_none() {
            return;
        }

        let task = self.server_start_task.as_ref().unwrap();
        if !task.is_finished() {
            return;
        }

        let task = self.server_start_task.take().unwrap();

        match task.await {
            Ok(Ok((server_handle, model_mgr, host, port))) => {
                // Populate loaded models list
                {
                    let mgr = model_mgr.lock().await;
                    self.server_panel_state.loaded_models = mgr
                        .list()
                        .iter()
                        .map(|m| crate::server_panel::LoadedModelInfo {
                            id: m.id.clone(),
                            name: m.model_info.name.clone(),
                            backend: m.backend_name.clone(),
                            is_default: mgr.default_id() == Some(m.id.as_str()),
                        })
                        .collect();
                }
                self.shared_model_manager = Some(model_mgr);

                self.server_panel_state.server_url = Some(format!("http://{}:{}", host, port));
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state.status_message = None;

                self.server_handle = Some(server_handle);
                self.log(&format!("Server started on {}:{}", host, port));

                // Update chat state to show server model is available
                if self.backend.is_none() {
                    let mgr = self.shared_model_manager.as_ref().unwrap();
                    let m = mgr.lock().await;
                    if let Some(default_id) = m.default_id() {
                        self.chat_state.current_model = Some(default_id.to_string());
                        if let Some(model) = m.list().iter().find(|lm| lm.id == default_id) {
                            self.chat_state.current_backend = Some(model.backend_name.clone());
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                self.server_panel_state.phase = ServerPhase::Error;
                self.server_panel_state.status_message =
                    Some(format!("Failed to start server: {}", e));
                self.log(&format!("Failed to start server: {}", e));
            }
            Err(e) => {
                self.server_panel_state.phase = ServerPhase::Error;
                self.server_panel_state.status_message =
                    Some(format!("Server start task crashed: {}", e));
                self.log(&format!("Server start task crashed: {}", e));
            }
        }
    }

    fn stop_server(&mut self) {
        if self.server_panel_state.phase == ServerPhase::Running {
            // Stop the detached server process by PID
            if let Some(ref state) = self.server_state {
                match server_manager::stop_by_pid(state.pid) {
                    Ok(()) => {}
                    Err(e) => {
                        self.server_panel_state.status_message =
                            Some(format!("Error stopping server: {}", e));
                    }
                }
            }
            // Also abort any in-process server handle (legacy path)
            if let Some(handle) = self.server_handle.take() {
                handle.abort();
            }
        } else if self.server_panel_state.phase == ServerPhase::LoadingModel {
            // Cancel the server start task (legacy in-process path)
            if let Some(task) = self.server_start_task.take() {
                task.abort();
            }
            // Cancel the health check polling task
            if let Some(task) = self.server_health_task.take() {
                task.abort();
            }
            // Kill the detached process if it was started
            if let Some(ref state) = self.server_state {
                let _ = server_manager::stop_by_pid(state.pid);
            }
        } else {
            self.server_panel_state.status_message = Some("Server is not running".to_string());
            return;
        }

        self.server_panel_state.phase = ServerPhase::Configuring;
        self.server_panel_state.server_url = None;
        self.server_panel_state.loaded_model_name = None;
        self.server_panel_state.loaded_backend_name = None;
        self.server_panel_state.loaded_models.clear();
        self.server_panel_state.unload_model_selected = 0;
        self.server_panel_state.default_model_selected = 0;
        self.shared_model_manager = None;
        self.server_state = None;
        self.server_panel_state.status_message = Some("Server stopped".to_string());
        self.log("Server stopped");

        // Clear chat model info if chat was using server model
        if self.backend.is_none() {
            self.chat_state.current_model = None;
            self.chat_state.current_backend = None;
        }
    }

    async fn load_additional_model(&mut self) {
        if self.server_panel_state.phase != ServerPhase::Running {
            self.server_panel_state.status_message = Some("Start the server first".to_string());
            return;
        }

        // Don't start if already loading
        if self.additional_model_load_task.is_some() {
            self.server_panel_state.status_message =
                Some("Already loading a model, please wait...".to_string());
            return;
        }

        let model_path = match self.server_panel_state.selected_model_path() {
            Some(p) => p,
            None => {
                self.server_panel_state.status_message =
                    Some("No model selected. Use Left/Right to pick a model.".to_string());
                return;
            }
        };

        let model_name = std::path::Path::new(&model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();

        self.server_panel_state.phase = ServerPhase::LoadingModel;
        self.server_panel_state.status_message =
            Some(format!("Loading additional model: {}...", model_name));
        self.additional_model_name_hint = Some(model_name.clone());

        // If detached server, use HTTP API to load model
        if let Some(ref state) = self.server_state {
            let host = state.host.clone();
            let port = state.port;
            let api_key = if self.server_panel_state.api_key.is_empty() {
                None
            } else {
                Some(self.server_panel_state.api_key.clone())
            };
            let gpu_layers = self.server_panel_state.gpu_layers;
            let context_size = self.server_panel_state.context_size;

            let task = tokio::spawn(async move {
                let client = reqwest::Client::new();
                let url = format!("http://{}:{}/v1/models/load", host, port);

                let body = serde_json::json!({
                    "model_path": model_path,
                    "gpu_layers": gpu_layers,
                    "context_size": context_size,
                    "set_default": false,
                });

                let mut req = client.post(&url).json(&body);
                if let Some(ref key) = api_key {
                    req = req.header("Authorization", format!("Bearer {}", key));
                }

                let resp = req
                    .send()
                    .await
                    .map_err(|e| format!("HTTP request failed: {}", e))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    return Err(format!("Server returned {}: {}", status, text));
                }

                // Parse response to get model_id
                let resp_json: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("Failed to parse response: {}", e))?;

                let model_id = resp_json
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                Ok::<String, String>(model_id)
            });

            let task = tokio::spawn(async move {
                AdditionalModelLoadResult::Detached(
                    task.await.unwrap_or(Err("Task panicked".to_string())),
                )
            });

            self.additional_model_load_task = Some(task);
            return;
        }

        // In-process server: use shared_model_manager
        let backend_type = self.server_panel_state.backend;
        let hardware = self.server_panel_state.hardware.clone();
        let load_config = self.server_panel_state.build_load_config(&model_path);

        let task = tokio::spawn(async move {
            let mut backend = BackendFactory::create(backend_type, &hardware).map_err(|e| {
                (
                    e,
                    std::path::Path::new(&load_config.model_path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model")
                        .to_string(),
                    "unknown".to_string(),
                )
            })?;
            backend.load_model(load_config).await.map_err(|e| {
                let name = backend
                    .model_info()
                    .map(|i| i.name.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                let bname = backend.name().to_string();
                (e, name, bname)
            })?;
            Ok::<Box<dyn Backend>, (athenas_core::AthenasError, String, String)>(backend)
        });

        let task = tokio::spawn(async move {
            AdditionalModelLoadResult::InProcess(task.await.unwrap_or(Err((
                athenas_core::AthenasError::Backend("Task panicked".to_string()),
                "unknown".to_string(),
                "unknown".to_string(),
            ))))
        });

        self.additional_model_load_task = Some(task);
    }

    async fn poll_additional_model_loading(&mut self) {
        if self.additional_model_load_task.is_none() {
            return;
        }

        let task = self.additional_model_load_task.as_ref().unwrap();
        if !task.is_finished() {
            return;
        }

        let task = self.additional_model_load_task.take().unwrap();
        let hint = self.additional_model_name_hint.take();

        match task.await {
            Ok(AdditionalModelLoadResult::InProcess(Ok(backend))) => {
                let model_name = backend
                    .model_info()
                    .map(|i| i.name.clone())
                    .unwrap_or_else(|| hint.unwrap_or_else(|| "unknown".to_string()));
                let backend_name = backend.name().to_string();

                if let Some(mgr) = &self.shared_model_manager {
                    let mut m = mgr.lock().await;
                    let model_id = m.add(backend);

                    self.server_panel_state.loaded_models = m
                        .list()
                        .iter()
                        .map(|lm| crate::server_panel::LoadedModelInfo {
                            id: lm.id.clone(),
                            name: lm.model_info.name.clone(),
                            backend: lm.backend_name.clone(),
                            is_default: m.default_id() == Some(lm.id.as_str()),
                        })
                        .collect();

                    self.server_panel_state.status_message = Some(format!(
                        "Loaded '{}' on {} (id: {})",
                        model_name, backend_name, model_id
                    ));

                    // Update chat state if chat has no local model
                    if self.backend.is_none() {
                        self.chat_state.current_model = Some(model_name);
                        self.chat_state.current_backend = Some(backend_name);
                    }
                }
                self.server_panel_state.phase = ServerPhase::Running;
            }
            Ok(AdditionalModelLoadResult::InProcess(Err((e, name, _)))) => {
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state.status_message =
                    Some(format!("Failed to load '{}': {}", name, e));
            }
            Ok(AdditionalModelLoadResult::Detached(Ok(model_id))) => {
                let model_name = hint.unwrap_or_else(|| "unknown".to_string());
                self.server_panel_state.status_message = Some(format!(
                    "Loaded '{}' on remote server (id: {})",
                    model_name, model_id
                ));
                // Fetch updated model list from detached server
                self.refresh_remote_loaded_models().await;
                self.server_panel_state.phase = ServerPhase::Running;
            }
            Ok(AdditionalModelLoadResult::Detached(Err(e))) => {
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state.status_message =
                    Some(format!("Failed to load model on remote server: {}", e));
            }
            Err(e) => {
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state.status_message =
                    Some(format!("Model loading task crashed: {}", e));
            }
        }
    }

    async fn unload_model_action(&mut self) {
        if self.server_panel_state.loaded_models.is_empty() {
            self.server_panel_state.status_message = Some("No models to unload".to_string());
            return;
        }

        let model_id = match self.server_panel_state.selected_unload_model_id() {
            Some(id) => id,
            None => return,
        };

        // Detached server: use HTTP API
        if let Some(ref state) = self.server_state {
            let host = state.host.clone();
            let port = state.port;
            let api_key = if self.server_panel_state.api_key.is_empty() {
                None
            } else {
                Some(self.server_panel_state.api_key.clone())
            };
            let mid = model_id.clone();

            let result = tokio::spawn(async move {
                let client = reqwest::Client::new();
                let url = format!("http://{}:{}/v1/models/unload", host, port);
                let body = serde_json::json!({ "model_id": mid });
                let mut req = client.post(&url).json(&body);
                if let Some(ref key) = api_key {
                    req = req.header("Authorization", format!("Bearer {}", key));
                }
                let resp = req
                    .send()
                    .await
                    .map_err(|e| format!("HTTP request failed: {}", e))?;
                if !resp.status().is_success() {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    return Err(format!("Server returned {}: {}", status, text));
                }
                Ok::<(), String>(())
            })
            .await;

            match result {
                Ok(Ok(())) => {
                    self.refresh_remote_loaded_models().await;
                    self.server_panel_state.status_message =
                        Some(format!("Unloaded model: {}", model_id));
                }
                Ok(Err(e)) => {
                    self.server_panel_state.status_message = Some(format!("Error: {}", e));
                }
                Err(e) => {
                    self.server_panel_state.status_message = Some(format!("Task crashed: {}", e));
                }
            }
            return;
        }

        // In-process server: use shared_model_manager
        if let Some(mgr) = &self.shared_model_manager {
            let mut m = mgr.lock().await;

            match m.remove(&model_id).await {
                Ok(()) => {
                    self.server_panel_state.loaded_models = m
                        .list()
                        .iter()
                        .map(|lm| crate::server_panel::LoadedModelInfo {
                            id: lm.id.clone(),
                            name: lm.model_info.name.clone(),
                            backend: lm.backend_name.clone(),
                            is_default: m.default_id() == Some(lm.id.as_str()),
                        })
                        .collect();

                    // Fix selection indices
                    if !self.server_panel_state.loaded_models.is_empty() {
                        if self.server_panel_state.unload_model_selected
                            >= self.server_panel_state.loaded_models.len()
                        {
                            self.server_panel_state.unload_model_selected =
                                self.server_panel_state.loaded_models.len() - 1;
                        }
                        if self.server_panel_state.default_model_selected
                            >= self.server_panel_state.loaded_models.len()
                        {
                            self.server_panel_state.default_model_selected =
                                self.server_panel_state.loaded_models.len() - 1;
                        }
                    }

                    self.server_panel_state.status_message =
                        Some(format!("Unloaded model: {}", model_id));
                }
                Err(e) => {
                    self.server_panel_state.status_message = Some(format!("Error: {}", e));
                }
            }
        }
    }

    async fn set_default_model_action(&mut self) {
        if self.server_panel_state.loaded_models.is_empty() {
            self.server_panel_state.status_message = Some("No models loaded".to_string());
            return;
        }

        let model_id = match self.server_panel_state.selected_default_model_id() {
            Some(id) => id,
            None => return,
        };

        // Detached server: use HTTP API
        if let Some(ref state) = self.server_state {
            let host = state.host.clone();
            let port = state.port;
            let api_key = if self.server_panel_state.api_key.is_empty() {
                None
            } else {
                Some(self.server_panel_state.api_key.clone())
            };
            let mid = model_id.clone();

            let result = tokio::spawn(async move {
                let client = reqwest::Client::new();
                let url = format!("http://{}:{}/v1/models/default", host, port);
                let body = serde_json::json!({ "model_id": mid });
                let mut req = client.post(&url).json(&body);
                if let Some(ref key) = api_key {
                    req = req.header("Authorization", format!("Bearer {}", key));
                }
                let resp = req
                    .send()
                    .await
                    .map_err(|e| format!("HTTP request failed: {}", e))?;
                if !resp.status().is_success() {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    return Err(format!("Server returned {}: {}", status, text));
                }
                Ok::<(), String>(())
            })
            .await;

            match result {
                Ok(Ok(())) => {
                    self.refresh_remote_loaded_models().await;
                    self.server_panel_state.status_message =
                        Some(format!("Default model set to: {}", model_id));
                }
                Ok(Err(e)) => {
                    self.server_panel_state.status_message = Some(format!("Error: {}", e));
                }
                Err(e) => {
                    self.server_panel_state.status_message = Some(format!("Task crashed: {}", e));
                }
            }
            return;
        }

        // In-process server: use shared_model_manager
        if let Some(mgr) = &self.shared_model_manager {
            let mut m = mgr.lock().await;
            match m.set_default(&model_id) {
                Ok(()) => {
                    self.server_panel_state.loaded_models = m
                        .list()
                        .iter()
                        .map(|lm| crate::server_panel::LoadedModelInfo {
                            id: lm.id.clone(),
                            name: lm.model_info.name.clone(),
                            backend: lm.backend_name.clone(),
                            is_default: m.default_id() == Some(lm.id.as_str()),
                        })
                        .collect();

                    self.server_panel_state.status_message =
                        Some(format!("Default model set to: {}", model_id));
                }
                Err(e) => {
                    self.server_panel_state.status_message = Some(format!("Error: {}", e));
                }
            }
        }
    }

    /// Fetch the list of loaded models from a detached server via GET /v1/models
    async fn refresh_remote_loaded_models(&mut self) {
        let Some(ref state) = self.server_state else {
            return;
        };
        let host = state.host.clone();
        let port = state.port;
        let api_key = if self.server_panel_state.api_key.is_empty() {
            None
        } else {
            Some(self.server_panel_state.api_key.clone())
        };

        let result = tokio::spawn(async move {
            let client = reqwest::Client::new();
            let url = format!("http://{}:{}/v1/models", host, port);
            let mut req = client.get(&url);
            if let Some(ref key) = api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            if !resp.status().is_success() {
                return Err(format!("Server returned {}", resp.status()));
            }
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))?;
            Ok::<serde_json::Value, String>(json)
        })
        .await;

        match result {
            Ok(Ok(json)) => {
                let mut models = Vec::new();
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    for m in data {
                        let id = m
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("?")
                            .to_string();
                        let name = m
                            .get("model_name")
                            .or_else(|| m.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("?")
                            .to_string();
                        let backend = m
                            .get("backend")
                            .and_then(|v| v.as_str())
                            .unwrap_or("remote")
                            .to_string();
                        let is_default = m
                            .get("is_default")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        models.push(crate::server_panel::LoadedModelInfo {
                            id,
                            name,
                            backend,
                            is_default,
                        });
                    }
                }
                self.server_panel_state.loaded_models = models;
            }
            Ok(Err(e)) => {
                tracing::warn!("Failed to fetch remote models: {}", e);
            }
            Err(e) => {
                tracing::warn!("Remote model fetch task crashed: {}", e);
            }
        }
    }

    async fn poll_server_status(&mut self) {
        if let Some(handle) = &mut self.server_handle {
            if handle.is_finished() {
                let result = handle.await;
                self.server_handle = None;

                match result {
                    Ok(Ok(())) => {
                        self.server_panel_state.phase = ServerPhase::Configuring;
                        self.server_panel_state.server_url = None;
                        self.server_panel_state.loaded_models.clear();
                        self.shared_model_manager = None;
                        self.server_panel_state.status_message = Some("Server stopped".to_string());
                        if self.backend.is_none() {
                            self.chat_state.current_model = None;
                            self.chat_state.current_backend = None;
                        }
                    }
                    Ok(Err(e)) => {
                        self.server_panel_state.phase = ServerPhase::Error;
                        self.server_panel_state.server_url = None;
                        self.server_panel_state.loaded_models.clear();
                        self.shared_model_manager = None;
                        self.server_panel_state.status_message =
                            Some(format!("Server error: {}", e));
                        if self.backend.is_none() {
                            self.chat_state.current_model = None;
                            self.chat_state.current_backend = None;
                        }
                    }
                    Err(_) => {
                        // Aborted — already handled by stop_server
                    }
                }
            }
        }
    }

    /// Handle key events for the API key modal.
    async fn handle_api_key_modal_key(&mut self, key: event::KeyEvent) {
        use crate::api_key_modal::ApiKeyModalPhase;

        match self.api_key_modal.phase.clone() {
            ApiKeyModalPhase::List => match key.code {
                KeyCode::Esc => {
                    self.api_key_modal.close();
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.api_key_modal.select_prev();
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.api_key_modal.select_next();
                }
                KeyCode::Enter => {
                    // Refresh key list
                    self.refresh_api_keys().await;
                }
                KeyCode::Char('n') => {
                    self.api_key_modal.start_create();
                }
                KeyCode::Char('r') => {
                    self.api_key_modal.start_revoke();
                }
                KeyCode::Char('d') => {
                    self.api_key_modal.start_delete();
                }
                _ => {}
            },

            ApiKeyModalPhase::CreateForm => match key.code {
                KeyCode::Esc => {
                    self.api_key_modal.cancel_create();
                }
                KeyCode::Enter => {
                    let submitted = self.api_key_modal.advance_form();
                    if submitted {
                        self.submit_create_api_key_modal().await;
                    }
                }
                KeyCode::Backspace => {
                    self.api_key_modal.form_backspace();
                }
                KeyCode::Char(c) => {
                    self.api_key_modal.form_input_char(c);
                }
                _ => {}
            },

            ApiKeyModalPhase::KeyRevealed => match key.code {
                KeyCode::Enter | KeyCode::Esc => {
                    self.api_key_modal.dismiss_revealed();
                    self.refresh_api_keys().await;
                }
                _ => {}
            },

            ApiKeyModalPhase::ConfirmRevoke => match key.code {
                KeyCode::Enter => {
                    if let Some(key_id) = self.api_key_modal.selected_key_id() {
                        let key_name = self
                            .api_key_modal
                            .selected_key()
                            .map(|k| k.name.clone())
                            .unwrap_or_default();
                        self.revoke_api_key_modal(key_id, key_name).await;
                    }
                }
                KeyCode::Esc => {
                    self.api_key_modal.cancel_confirm();
                }
                _ => {}
            },

            ApiKeyModalPhase::ConfirmDelete => match key.code {
                KeyCode::Enter => {
                    if let Some(key_id) = self.api_key_modal.selected_key_id() {
                        let key_name = self
                            .api_key_modal
                            .selected_key()
                            .map(|k| k.name.clone())
                            .unwrap_or_default();
                        self.delete_api_key_modal(key_id, key_name).await;
                    }
                }
                KeyCode::Esc => {
                    self.api_key_modal.cancel_confirm();
                }
                _ => {}
            },
        }
    }

    /// Submit the API key creation request from the modal.
    async fn submit_create_api_key_modal(&mut self) {
        let Some(ref state) = self.server_state else {
            self.api_key_modal
                .set_error("Server not running".to_string());
            return;
        };
        let host = state.host.clone();
        let port = state.port;
        let admin_key = if self.server_panel_state.api_key.is_empty() {
            None
        } else {
            Some(self.server_panel_state.api_key.clone())
        };
        let (name, rate_limit, token_limit, allowed_models) = self.api_key_modal.form_values();

        let result = tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .map_err(|e| e.to_string())?;
            let url = format!("http://{}:{}/v1/keys", host, port);
            let body = serde_json::json!({
                "name": name,
                "rate_limit_per_minute": rate_limit,
                "daily_token_limit": token_limit,
                "allowed_models": allowed_models,
            });
            let mut req = client.post(&url).json(&body);
            if let Some(ref key) = admin_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            if !resp.status().is_success() {
                return Err(format!("Server returned {}", resp.status()));
            }
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))?;
            Ok::<serde_json::Value, String>(json)
        })
        .await;

        match result {
            Ok(Ok(json)) => {
                let full_key = json
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
                let key_name = json
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string();
                self.api_key_modal.reveal_key(key_name, full_key);
            }
            Ok(Err(e)) => {
                self.api_key_modal
                    .set_error(format!("Failed to create key: {}", e));
            }
            Err(e) => {
                self.api_key_modal
                    .set_error(format!("Request failed: {}", e));
            }
        }
    }

    /// Revoke an API key from the modal.
    async fn revoke_api_key_modal(&mut self, key_id: String, key_name: String) {
        let Some(ref state) = self.server_state else {
            self.api_key_modal
                .set_error("Server not running".to_string());
            return;
        };
        let host = state.host.clone();
        let port = state.port;
        let admin_key = if self.server_panel_state.api_key.is_empty() {
            None
        } else {
            Some(self.server_panel_state.api_key.clone())
        };
        let kid = key_id.clone();

        let result = tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .map_err(|e| e.to_string())?;
            let url = format!("http://{}:{}/v1/keys/{}/revoke", host, port, kid);
            let mut req = client.post(&url);
            if let Some(ref key) = admin_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            if !resp.status().is_success() {
                return Err(format!("Server returned {}", resp.status()));
            }
            Ok::<(), String>(())
        })
        .await;

        match result {
            Ok(Ok(())) => {
                self.api_key_modal.cancel_confirm();
                self.api_key_modal
                    .set_info(format!("Revoked key: {}", key_name));
                self.refresh_api_keys().await;
            }
            Ok(Err(e)) => {
                self.api_key_modal
                    .set_error(format!("Failed to revoke: {}", e));
            }
            Err(e) => {
                self.api_key_modal
                    .set_error(format!("Request failed: {}", e));
            }
        }
    }

    /// Delete an API key from the modal.
    async fn delete_api_key_modal(&mut self, key_id: String, key_name: String) {
        let Some(ref state) = self.server_state else {
            self.api_key_modal
                .set_error("Server not running".to_string());
            return;
        };
        let host = state.host.clone();
        let port = state.port;
        let admin_key = if self.server_panel_state.api_key.is_empty() {
            None
        } else {
            Some(self.server_panel_state.api_key.clone())
        };
        let kid = key_id.clone();

        let result = tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .map_err(|e| e.to_string())?;
            let url = format!("http://{}:{}/v1/keys/{}", host, port, kid);
            let mut req = client.delete(&url);
            if let Some(ref key) = admin_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            if !resp.status().is_success() {
                return Err(format!("Server returned {}", resp.status()));
            }
            Ok::<(), String>(())
        })
        .await;

        match result {
            Ok(Ok(())) => {
                self.api_key_modal.cancel_confirm();
                self.api_key_modal
                    .set_info(format!("Deleted key: {}", key_name));
                self.refresh_api_keys().await;
            }
            Ok(Err(e)) => {
                self.api_key_modal
                    .set_error(format!("Failed to delete: {}", e));
            }
            Err(e) => {
                self.api_key_modal
                    .set_error(format!("Request failed: {}", e));
            }
        }
    }

    async fn refresh_api_keys(&mut self) {
        let Some(ref state) = self.server_state else {
            self.server_panel_state.status_message =
                Some("Server not running — start it first".to_string());
            return;
        };
        let host = state.host.clone();
        let port = state.port;
        let api_key = if self.server_panel_state.api_key.is_empty() {
            None
        } else {
            Some(self.server_panel_state.api_key.clone())
        };

        let result = tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .map_err(|e| e.to_string())?;
            let url = format!("http://{}:{}/v1/keys", host, port);
            let mut req = client.get(&url);
            if let Some(ref key) = api_key {
                req = req.header("Authorization", format!("Bearer {}", key));
            }
            let resp = req
                .send()
                .await
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            if !resp.status().is_success() {
                return Err(format!("Server returned {}", resp.status()));
            }
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| format!("Failed to parse response: {}", e))?;
            Ok::<serde_json::Value, String>(json)
        })
        .await;

        match result {
            Ok(Ok(json)) => {
                let mut keys = Vec::new();
                if let Some(arr) = json.get("keys").and_then(|k| k.as_array()) {
                    for k in arr {
                        keys.push(crate::server_panel::ApiKeyInfo {
                            key_id: k
                                .get("key_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                                .to_string(),
                            api_key: k
                                .get("api_key")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                                .to_string(),
                            name: k
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                                .to_string(),
                            active: k.get("active").and_then(|v| v.as_bool()).unwrap_or(true),
                            rate_limit_per_minute: k
                                .get("rate_limit_per_minute")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                as u32,
                            daily_token_limit: k
                                .get("daily_token_limit")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            allowed_models: k
                                .get("allowed_models")
                                .and_then(|v| v.as_array())
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                            created_at: k
                                .get("created_at")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                                .to_string(),
                        });
                    }
                }
                let count = keys.len();
                self.server_panel_state.api_keys = keys.clone();
                if self.server_panel_state.api_key_selected
                    >= self.server_panel_state.api_keys.len()
                {
                    self.server_panel_state.api_key_selected = 0;
                }
                self.server_panel_state.status_message =
                    Some(format!("Loaded {} API key(s)", count));
                // Also update the modal if it's open
                self.api_key_modal.set_keys(keys);
            }
            Ok(Err(e)) => {
                let msg = format!("Failed to fetch API keys: {}", e);
                self.server_panel_state.status_message = Some(msg.clone());
                if self.api_key_modal.open {
                    self.api_key_modal.set_error(msg);
                }
            }
            Err(e) => {
                let msg = format!("Request failed: {}", e);
                self.server_panel_state.status_message = Some(msg.clone());
                if self.api_key_modal.open {
                    self.api_key_modal.set_error(msg);
                }
            }
        }
    }
}
