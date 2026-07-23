use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;

use athenas_core::{AppConfig, HardwareInfo, ModelRegistry, Result};
use athenas_inference::{
    Backend, BackendFactory, ChatMessage, ChatRequest, MessageContent, ModelLoadConfig, Role,
    StreamChunk,
};

use crate::chat::ChatState;
use crate::components;
use crate::model_browser::{BrowserPhase, ModelBrowserState};
use crate::model_list::ModelListState;
use crate::server_panel::{ConfigField, ServerPanelState, ServerPhase};
use crate::settings::SettingsState;

type AdditionalModelLoadResult =
    std::result::Result<Box<dyn Backend>, (athenas_core::AthenasError, String, String)>;

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
    // Background chat streaming state
    chat_stream_rx: Option<tokio::sync::mpsc::Receiver<StreamChunk>>,
}

impl TuiApp {
    pub fn new(config: AppConfig, hardware: HardwareInfo) -> Self {
        let registry = ModelRegistry::new(config.paths.models_dir.clone());
        let models = registry.list_local_models().unwrap_or_default();

        let mut model_list_state = ModelListState::default();
        model_list_state.set_models(models);

        let settings_state = SettingsState::new(config.clone());
        let server_panel_state = ServerPanelState::new(&config, hardware.clone());

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
            is_loading_model: false,
            model_load_task: None,
            loading_spinner: 0,
            additional_model_load_task: None,
            additional_model_name_hint: None,
            server_start_task: None,
            logs_state: crate::log_buffer::LogsState::new(crate::log_buffer::LogBuffer::new(500)),
            chat_stream_rx: None,
        }
    }

    pub fn with_log_buffer(
        config: AppConfig,
        hardware: HardwareInfo,
        log_buffer: crate::log_buffer::LogBuffer,
    ) -> Self {
        let mut app = Self::new(config, hardware);
        app.logs_state = crate::log_buffer::LogsState::new(log_buffer);
        app
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
            // Poll background download progress (non-blocking)
            self.poll_download_progress().await;

            // Poll server task status
            self.poll_server_status().await;

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

                    // Global Tab cycling (skip when editing or in chat mode
                    // where Tab is used for thinking toggle)
                    if key.code == KeyCode::Tab
                        && self.mode != AppMode::Chat
                        && !(self.mode == AppMode::Settings && self.settings_state.editing)
                        && !(self.mode == AppMode::Server && self.server_panel_state.editing)
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
                components::render_chat_area(
                    f,
                    content,
                    &mut self.chat_state,
                    self.is_loading_model,
                    self.loading_spinner,
                );
            }
            AppMode::ModelList => {
                components::render_model_list(f, content, &self.model_list_state);
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
                    self.settings_state.start_edit();
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
            KeyCode::Char('c') => {
                self.logs_state.clear();
            }
            KeyCode::Char('a') => {
                self.logs_state.auto_scroll = !self.logs_state.auto_scroll;
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
                } else {
                    self.chat_state.add_message(
                        "system",
                        "No model loaded. Press F2 to select a model, or start the server (F4) to load one.",
                    );
                    return;
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
        };

        // Get a backend reference: prefer local backend, fall back to server's model manager
        let backend_ref: Option<Box<dyn Backend>> = if let Some(ref b) = self.backend {
            Some(b.boxed_clone())
        } else if let Some(mgr) = &self.shared_model_manager {
            let m = mgr.lock().await;
            m.get(None).map(|b| b.boxed_clone())
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
            context_size,
            batch_size,
            threads,
            flash_attention: self.config.inference.flash_attention,
            use_mmap: true,
            use_mlock: false,
            reasoning_enabled: self.config.inference.reasoning_enabled,
            reasoning_budget: self.config.inference.reasoning_budget,
            mmproj_path: None,
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
                    if let Some(ref i) = info {
                        self.chat_state.current_model = Some(i.name.clone());
                        self.chat_state.current_backend = Some(i.backend_name.clone());
                    }
                    self.chat_state
                        .add_message("system", "Model loaded successfully!");
                    self.backend = Some(backend);
                    self.is_loading_model = false;
                }
                Ok(Err(e)) => {
                    self.chat_state
                        .add_message("system", &format!("Failed to load model: {}", e));
                    self.is_loading_model = false;
                }
                Err(e) => {
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

        self.server_panel_state.phase = ServerPhase::LoadingModel;
        self.server_panel_state.status_message = Some(format!("Loading model: {}...", model_path));

        let backend_type = self.server_panel_state.backend;
        let hardware = self.server_panel_state.hardware.clone();
        let load_config = self.server_panel_state.build_load_config(&model_path);
        let server_config = self.server_panel_state.build_app_config(&self.config);
        let host = self.server_panel_state.host.clone();
        let port = self.server_panel_state.port;

        let task = tokio::spawn(async move {
            let mut backend = BackendFactory::create(backend_type, &hardware)?;
            backend.load_model(load_config).await?;

            let api_server = athenas_server::ApiServer::new(server_config, backend);
            let model_mgr = api_server.model_manager();

            // Start the server in the background
            let server_host = host.clone();
            let server_port = port;
            let server_handle =
                tokio::spawn(async move { api_server.start(&server_host, server_port).await });

            Ok::<
                (
                    tokio::task::JoinHandle<athenas_core::Result<()>>,
                    athenas_server::SharedModelManager,
                    String,
                    u16,
                ),
                athenas_core::AthenasError,
            >((server_handle, model_mgr, host, port))
        });

        self.server_start_task = Some(task);
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
            }
            Err(e) => {
                self.server_panel_state.phase = ServerPhase::Error;
                self.server_panel_state.status_message =
                    Some(format!("Server start task crashed: {}", e));
            }
        }
    }

    fn stop_server(&mut self) {
        if self.server_panel_state.phase == ServerPhase::Running {
            // Abort the server task
            if let Some(handle) = self.server_handle.take() {
                handle.abort();
            }
        } else if self.server_panel_state.phase == ServerPhase::LoadingModel {
            // Cancel the server start task
            if let Some(task) = self.server_start_task.take() {
                task.abort();
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
        self.server_panel_state.status_message = Some("Server stopped".to_string());

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
            Ok(Ok(backend)) => {
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
            Ok(Err((e, name, _))) => {
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state.status_message =
                    Some(format!("Failed to load '{}': {}", name, e));
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
}
