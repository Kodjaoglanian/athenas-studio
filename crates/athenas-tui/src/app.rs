use std::io::stdout;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use athenas_core::{AppConfig, HardwareInfo, ModelInfo, ModelRegistry, Result};
use athenas_inference::{
    Backend, BackendFactory, ChatMessage, ChatRequest, MessageContent, ModelLoadConfig,
    RemoteBackend, Role, StreamChunk,
};

use crate::chat::ChatState;
use crate::components;
use crate::log_buffer::{LogBuffer, LogLevelFilter, LogsState};
use crate::model_browser::{BrowserPhase, ModelBrowserState};
use crate::model_list::ModelListState;
use crate::server_manager;
use crate::server_panel::{ConfigField, ServerPanelState, ServerPhase};
use crate::settings::SettingsState;
use crate::theme;

enum ChatStreamEvent {
    Chunk(StreamChunk),
    Error(String),
}

enum AdditionalModelLoadResult {
    InProcess(std::result::Result<Box<dyn Backend>, (athenas_core::AthenasError, String, String)>),
    Detached(std::result::Result<String, String>),
}

type AdditionalModelLoadTask = Option<tokio::task::JoinHandle<AdditionalModelLoadResult>>;
type BrowserSearchTask = Option<
    tokio::task::JoinHandle<std::result::Result<Vec<athenas_hub::ModelSearchResult>, String>>,
>;
type BrowserFilesTask = Option<
    tokio::task::JoinHandle<
        std::result::Result<Vec<(String, Option<u64>, Option<String>)>, String>,
    >,
>;
type ModelsRefreshTask = Option<tokio::task::JoinHandle<Vec<ModelInfo>>>;
type ModelDeleteTask = Option<tokio::task::JoinHandle<(String, std::result::Result<(), String>)>>;

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

fn http_client(timeout_secs: u64) -> std::result::Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))
}

fn new_chat_input() -> tui_textarea::TextArea<'static> {
    let mut ta = tui_textarea::TextArea::default();
    ta.set_block(theme::rounded_block(" ✦ Message ", true));
    ta.set_placeholder_text("Message Athenas... (Enter send · Alt+Enter newline · /help)");
    ta.set_placeholder_style(ratatui::style::Style::default().fg(theme::TEXT_MUTED));
    ta.set_cursor_line_style(ratatui::style::Style::default());
    ta
}

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

    pub fn from_tab_index(idx: usize) -> Self {
        match idx {
            0 => AppMode::Chat,
            1 => AppMode::ModelList,
            2 => AppMode::Browser,
            3 => AppMode::Server,
            4 => AppMode::Settings,
            _ => AppMode::Logs,
        }
    }
}

pub struct TuiApp {
    config: AppConfig,
    hardware: HardwareInfo,
    chat_state: ChatState,
    /// Multi-line text input for chat (tui-textarea)
    chat_input: tui_textarea::TextArea<'static>,
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
    // Non-blocking HuggingFace search & file listing tasks
    browser_search_task: BrowserSearchTask,
    browser_files_task: BrowserFilesTask,
    // Non-blocking local model scan & delete tasks
    models_refresh_task: ModelsRefreshTask,
    model_delete_task: ModelDeleteTask,
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
    logs_state: LogsState,
    // Background server log tailer task + stop flag
    server_log_tailer: Option<tokio::task::JoinHandle<()>>,
    log_tailer_stop: Arc<AtomicBool>,
    // Background chat streaming state
    chat_stream_rx: Option<tokio::sync::mpsc::Receiver<ChatStreamEvent>>,
    // Cancellation token for the current chat generation.
    chat_cancel: Option<tokio::sync::watch::Sender<bool>>,
    // API Key management modal
    api_key_modal: crate::api_key_modal::ApiKeyModalState,
    // Global Help modal
    help_open: bool,
    // Periodic system memory refresh
    mem_refresh_task: Option<tokio::task::JoinHandle<(u64, u64)>>,
    last_mem_refresh: std::time::Instant,
}

impl TuiApp {
    pub fn new(config: AppConfig, hardware: HardwareInfo) -> Self {
        let log_buffer = LogBuffer::new(2000);
        Self::build_internal(config, hardware, log_buffer)
    }

    pub fn with_log_buffer(
        config: AppConfig,
        hardware: HardwareInfo,
        log_buffer: LogBuffer,
    ) -> Self {
        Self::build_internal(config, hardware, log_buffer)
    }

    fn build_internal(config: AppConfig, hardware: HardwareInfo, log_buffer: LogBuffer) -> Self {
        let registry = ModelRegistry::new(config.paths.models_dir.clone());
        let models = registry.list_local_models().unwrap_or_default();

        let mut model_list_state = ModelListState::default();
        model_list_state.set_models(models.clone());

        let settings_state = SettingsState::new(config.clone()).with_hardware(hardware.clone());
        let mut server_panel_state = ServerPanelState::new(&config, hardware.clone());
        server_panel_state.set_models(models);

        // Spawn a background health check to detect an already-running server
        let health_task = tokio::spawn(server_manager::check_running());

        // Spawn a SINGLE background log tailer with an explicit stop flag
        let stop_flag = Arc::new(AtomicBool::new(false));
        let tailer_stop = Arc::clone(&stop_flag);
        let tailer_buffer = log_buffer.clone();
        let log_tailer = tokio::task::spawn_blocking(move || {
            Self::run_server_log_tailer(tailer_buffer, tailer_stop)
        });

        Self {
            config,
            hardware,
            chat_state: ChatState::default(),
            chat_input: new_chat_input(),
            model_list_state,
            browser_state: ModelBrowserState::default(),
            server_panel_state,
            settings_state,
            mode: AppMode::Chat,
            backend: None,
            download_progress_rx: None,
            download_task: None,
            browser_search_task: None,
            browser_files_task: None,
            models_refresh_task: None,
            model_delete_task: None,
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
            logs_state: LogsState::new(log_buffer),
            server_log_tailer: Some(log_tailer),
            log_tailer_stop: stop_flag,
            chat_stream_rx: None,
            chat_cancel: None,
            api_key_modal: crate::api_key_modal::ApiKeyModalState::new(),
            help_open: false,
            mem_refresh_task: None,
            // Start stale so the first server-tab visit refreshes immediately
            last_mem_refresh: std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(60))
                .unwrap_or_else(std::time::Instant::now),
        }
    }

    /// Log a TUI event to the tracing system (appears in the logs page).
    fn log(&self, msg: &str) {
        tracing::info!("{}", msg);
    }

    /// Port encoded in `llama-server-<port>-<id>.log`.
    fn llama_log_port(name: &str) -> Option<&str> {
        let rest = name.strip_prefix("llama-server-")?;
        let port = rest.split('-').next()?;
        if port.chars().all(|c| c.is_ascii_digit()) && !port.is_empty() {
            Some(port)
        } else {
            None
        }
    }

    /// Ports of llama-server processes that are running now.
    /// Returns `None` on platforms without `/proc` (macOS/Windows) so the caller
    /// can fall back to tailing any active log file that grows.
    fn live_llama_server_ports() -> Option<std::collections::HashSet<String>> {
        let entries = std::fs::read_dir("/proc").ok()?;
        let mut ports = std::collections::HashSet::new();
        for entry in entries.flatten() {
            let pid = entry.file_name();
            let pid = pid.to_string_lossy();
            if !pid.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let Ok(raw) = std::fs::read(format!("/proc/{pid}/cmdline")) else {
                continue;
            };
            let text = String::from_utf8_lossy(&raw).replace('\0', " ");
            if !text.contains("llama-server") {
                continue;
            }
            let parts: Vec<&str> = text.split_whitespace().collect();
            if let Some(i) = parts.iter().position(|arg| *arg == "--port") {
                if let Some(port) = parts.get(i + 1) {
                    ports.insert((*port).to_string());
                }
            }
        }
        Some(ports)
    }

    /// Background blocking thread that tails `~/.athenas/server.log` and active
    /// `llama-server-*.log` files into the in-memory `LogBuffer`.
    fn run_server_log_tailer(buffer: LogBuffer, stop: Arc<AtomicBool>) {
        let log_path = match dirs::home_dir() {
            Some(h) => h.join(".athenas").join("server.log"),
            None => return,
        };
        let llama_log_dir = match dirs::home_dir() {
            Some(h) => h.join(".athenas"),
            None => return,
        };

        use std::io::{BufRead, BufReader, Seek, SeekFrom};

        let mut last_size: u64 = 0;
        let mut llama_last_sizes = std::collections::HashMap::new();
        let mut initialized = false;
        let mut live_ports = Self::live_llama_server_ports();
        let mut last_proc_scan = std::time::Instant::now();

        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if stop.load(Ordering::Relaxed) {
                break;
            }

            // --- Tail server.log (athenas serve output) ---
            let current_size = match std::fs::metadata(&log_path) {
                Ok(m) => m.len(),
                Err(_) => {
                    initialized = false;
                    last_size = 0;
                    0
                }
            };

            if current_size > 0 {
                if !initialized {
                    initialized = true;
                    if current_size > 50_000 {
                        last_size = current_size - 50_000;
                    } else {
                        last_size = 0;
                    }
                }

                if current_size < last_size {
                    last_size = 0;
                }

                if current_size != last_size {
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

            // Throttle /proc scan to once every 2 seconds
            if last_proc_scan.elapsed() >= std::time::Duration::from_secs(2) {
                live_ports = Self::live_llama_server_ports();
                last_proc_scan = std::time::Instant::now();
            }

            let Ok(entries) = std::fs::read_dir(&llama_log_dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !(name == "llama-server.log"
                    || (name.starts_with("llama-server-") && name.ends_with(".log")))
                {
                    continue;
                }
                if let (Some(ref ports), Some(port)) = (&live_ports, Self::llama_log_port(name)) {
                    if !ports.contains(port) {
                        continue;
                    }
                }

                let Ok(metadata) = entry.metadata() else {
                    continue;
                };
                let size = metadata.len();
                let last_size = llama_last_sizes.entry(path.clone()).or_insert(size);
                if size < *last_size {
                    *last_size = 0;
                }
                if size == *last_size {
                    continue;
                }

                if let Ok(mut f) = std::fs::File::open(&path) {
                    let _ = f.seek(SeekFrom::Start(*last_size));
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
                    *last_size = size;
                }
            }
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        // Install a panic hook that restores the terminal before printing the panic
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let mut out = stdout();
            let _ = execute!(
                out,
                DisableMouseCapture,
                DisableBracketedPaste,
                LeaveAlternateScreen
            );
            default_hook(info);
        }));

        enable_raw_mode().map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;
        let mut stdout = stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )
        .map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal =
            Terminal::new(backend).map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;

        let result = self.main_loop(&mut terminal).await;

        // Signal the background log tailer to stop cleanly
        self.log_tailer_stop.store(true, Ordering::Relaxed);
        if let Some(tailer) = self.server_log_tailer.take() {
            tailer.abort();
        }

        let _ = disable_raw_mode();
        let _ = execute!(
            terminal.backend_mut(),
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = terminal.show_cursor();

        result
    }

    fn is_modal_or_edit_active(&self) -> bool {
        self.api_key_modal.open
            || self.help_open
            || self.model_list_state.confirm_delete.is_some()
            || (self.mode == AppMode::Settings && self.settings_state.editing)
            || (self.mode == AppMode::Server && self.server_panel_state.editing)
            || (self.mode == AppMode::ModelList && self.model_list_state.filtering)
            || (self.mode == AppMode::Logs && self.logs_state.searching)
            || (self.mode == AppMode::Browser && self.browser_state.phase == BrowserPhase::Search)
    }

    fn switch_to_mode(&mut self, mode: AppMode) {
        self.mode = mode;
        if matches!(self.mode, AppMode::ModelList | AppMode::Server) {
            self.spawn_refresh_models();
        }
    }

    fn spawn_refresh_models(&mut self) {
        if self.models_refresh_task.is_some() {
            return;
        }
        let models_dir = self.config.paths.models_dir.clone();
        self.models_refresh_task = Some(tokio::task::spawn_blocking(move || {
            let registry = ModelRegistry::new(models_dir);
            registry.list_local_models().unwrap_or_default()
        }));
    }

    async fn poll_models_refresh(&mut self) {
        let Some(task) = self.models_refresh_task.take() else {
            return;
        };
        if !task.is_finished() {
            self.models_refresh_task = Some(task);
            return;
        }
        if let Ok(models) = task.await {
            self.model_list_state.set_models(models.clone());
            self.server_panel_state.set_models(models);
        }
    }

    async fn poll_model_delete(&mut self) {
        let Some(task) = self.model_delete_task.take() else {
            return;
        };
        if !task.is_finished() {
            self.model_delete_task = Some(task);
            return;
        }
        if let Ok((name, res)) = task.await {
            match res {
                Ok(()) => {
                    self.model_list_state
                        .set_status(format!("Deleted model '{}'", name));
                    self.chat_state
                        .add_message("system", &format!("Model '{}' deleted.", name));
                    self.spawn_refresh_models();
                }
                Err(e) => {
                    self.model_list_state
                        .set_error(format!("Failed to delete '{}': {}", name, e));
                    self.chat_state
                        .add_error_message(&format!("Failed to delete model '{}': {}", name, e));
                }
            }
        }
    }

    async fn save_config_async(&self, config: AppConfig) -> std::result::Result<(), String> {
        tokio::task::spawn_blocking(move || config.save().map_err(|e| e.to_string()))
            .await
            .map_err(|e| format!("Save task failed: {}", e))?
    }

    fn sync_config_from_settings(&mut self) {
        self.config = self.settings_state.config.clone();
        self.server_panel_state.sync_from_config(&self.config);
    }

    fn sync_config_from_server_panel(&mut self) -> AppConfig {
        let cfg = self.server_panel_state.build_app_config(&self.config);
        self.config = cfg.clone();
        self.settings_state.config = cfg.clone();
        cfg
    }

    fn scroll_chat_up(&mut self, step: usize) {
        if self.chat_state.auto_scroll {
            self.chat_state.auto_scroll = false;
            self.chat_state.scroll = self.chat_state.max_scroll.saturating_sub(step);
        } else {
            self.chat_state.scroll = self.chat_state.scroll.saturating_sub(step);
        }
    }

    fn scroll_chat_down(&mut self, step: usize) {
        if !self.chat_state.auto_scroll {
            self.chat_state.scroll = self.chat_state.scroll.saturating_add(step);
            if self.chat_state.scroll >= self.chat_state.max_scroll {
                self.chat_state.auto_scroll = true;
            }
        }
    }

    async fn main_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> Result<()> {
        loop {
            // Poll all non-blocking background tasks
            self.poll_download_progress().await;
            self.poll_browser_search().await;
            self.poll_browser_files().await;
            self.poll_models_refresh().await;
            self.poll_model_delete().await;
            self.poll_server_status().await;
            self.poll_server_health().await;
            self.poll_model_loading().await;
            self.poll_additional_model_loading().await;
            self.poll_server_start_task().await;
            self.poll_mem_refresh().await;
            self.poll_chat_stream().await;

            // Smooth 10fps spinner tick (100ms loop interval)
            self.loading_spinner = self.loading_spinner.wrapping_add(1);

            terminal.draw(|f| self.render(f)).ok();

            if event::poll(std::time::Duration::from_millis(100))
                .map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?
            {
                let ev =
                    event::read().map_err(|e| athenas_core::AthenasError::Tui(e.to_string()))?;

                match ev {
                    Event::Paste(text) => {
                        self.handle_paste(&text);
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse(mouse);
                    }
                    Event::Key(key) => {
                        if key.kind != KeyEventKind::Press {
                            continue;
                        }

                        // Global quit
                        if key.code == KeyCode::Char('c')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            break;
                        }

                        // Help modal intercepts keys when open
                        if self.help_open {
                            if matches!(
                                key.code,
                                KeyCode::Esc
                                    | KeyCode::Char('?')
                                    | KeyCode::Enter
                                    | KeyCode::Char('q')
                            ) {
                                self.help_open = false;
                            }
                            continue;
                        }

                        // API key modal intercepts all keys when open
                        if self.api_key_modal.open {
                            self.handle_api_key_modal_key(key).await;
                            continue;
                        }

                        // Delete confirmation modal on Models tab intercepts keys
                        if self.mode == AppMode::ModelList
                            && self.model_list_state.confirm_delete.is_some()
                        {
                            self.handle_model_list_key(key).await;
                            continue;
                        }

                        let modal_or_edit = self.is_modal_or_edit_active();

                        // Global F1-F6 & Alt+1..Alt+6 tab switching (blocked during modals/edits)
                        if !modal_or_edit {
                            let target_tab = match key.code {
                                KeyCode::F(1) => Some(0),
                                KeyCode::F(2) => Some(1),
                                KeyCode::F(3) => Some(2),
                                KeyCode::F(4) => Some(3),
                                KeyCode::F(5) => Some(4),
                                KeyCode::F(6) => Some(5),
                                KeyCode::Char(c @ '1'..='6')
                                    if key.modifiers.contains(KeyModifiers::ALT) =>
                                {
                                    Some((c as u8 - b'1') as usize)
                                }
                                KeyCode::Char(c @ '1'..='6')
                                    if self.mode != AppMode::Chat && key.modifiers.is_empty() =>
                                {
                                    Some((c as u8 - b'1') as usize)
                                }
                                _ => None,
                            };

                            if let Some(tab) = target_tab {
                                self.switch_to_mode(AppMode::from_tab_index(tab));
                                continue;
                            }

                            // '?' opens Help modal (when not in Chat text input)
                            if key.code == KeyCode::Char('?') && self.mode != AppMode::Chat {
                                self.help_open = true;
                                continue;
                            }

                            // Tab / Shift+Tab global tab cycling (skip Chat where Tab toggles thinking)
                            if self.mode != AppMode::Chat {
                                if key.code == KeyCode::Tab {
                                    let next = (self.mode.tab_index() + 1) % 6;
                                    self.switch_to_mode(AppMode::from_tab_index(next));
                                    continue;
                                }
                                if key.code == KeyCode::BackTab {
                                    let prev = (self.mode.tab_index() + 5) % 6;
                                    self.switch_to_mode(AppMode::from_tab_index(prev));
                                    continue;
                                }
                            }
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
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn handle_paste(&mut self, text: &str) {
        if self.help_open || self.api_key_modal.open {
            return;
        }
        match self.mode {
            AppMode::Chat => {
                // Insert multiline text cleanly without triggering Enter/send
                self.chat_input.insert_str(text);
                self.chat_state.auto_scroll = true;
            }
            AppMode::Browser if self.browser_state.phase == BrowserPhase::Search => {
                self.browser_state
                    .search_input
                    .push_str(text.replace(['\r', '\n'], " ").trim());
            }
            AppMode::Settings if self.settings_state.editing => {
                self.settings_state
                    .edit_buffer
                    .push_str(text.trim_matches(['\r', '\n']));
            }
            AppMode::Server if self.server_panel_state.editing => {
                self.server_panel_state
                    .edit_buffer
                    .push_str(text.trim_matches(['\r', '\n']));
            }
            AppMode::ModelList if self.model_list_state.filtering => {
                self.model_list_state
                    .filter
                    .push_str(text.trim_matches(['\r', '\n']));
                self.model_list_state.clamp_selection();
            }
            AppMode::Logs if self.logs_state.searching => {
                self.logs_state
                    .search_query
                    .push_str(text.trim_matches(['\r', '\n']));
            }
            _ => {}
        }
    }

    fn handle_mouse(&mut self, mouse: event::MouseEvent) {
        if self.help_open || self.api_key_modal.open {
            return;
        }
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                // Click on top header row (row 0) switches tabs
                if mouse.row == 0 && !self.is_modal_or_edit_active() {
                    if let Some(tab) = components::header_tab_at_col(mouse.column) {
                        self.switch_to_mode(AppMode::from_tab_index(tab));
                        return;
                    }
                }
                // Click on Server Panel sub-tab bar (row 8: header=1 + telemetry=7)
                if self.mode == AppMode::Server
                    && mouse.row == 8
                    && !self.server_panel_state.editing
                {
                    if let Some(subtab) = components::server_subtab_at_col(mouse.column) {
                        self.server_panel_state.set_subtab(subtab);
                    }
                }
            }
            MouseEventKind::ScrollUp => match self.mode {
                AppMode::Chat => self.scroll_chat_up(3),
                AppMode::ModelList => self.model_list_state.previous(),
                AppMode::Browser => match self.browser_state.phase {
                    BrowserPhase::Results => self.browser_state.prev_result(),
                    BrowserPhase::SelectFile => self.browser_state.prev_file(),
                    _ => {}
                },
                AppMode::Server => self.server_panel_state.previous(),
                AppMode::Settings => self.settings_state.previous(),
                AppMode::Logs => self.logs_state.scroll_up(3),
            },
            MouseEventKind::ScrollDown => match self.mode {
                AppMode::Chat => self.scroll_chat_down(3),
                AppMode::ModelList => self.model_list_state.next(),
                AppMode::Browser => match self.browser_state.phase {
                    BrowserPhase::Results => self.browser_state.next_result(),
                    BrowserPhase::SelectFile => self.browser_state.next_file(),
                    _ => {}
                },
                AppMode::Server => self.server_panel_state.next(),
                AppMode::Settings => self.settings_state.next(),
                AppMode::Logs => self.logs_state.scroll_down(3),
            },
            _ => {}
        }
    }

    fn render(&mut self, f: &mut ratatui::Frame) {
        let area = f.area();

        let chunks = ratatui::layout::Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([
                ratatui::layout::Constraint::Length(1),
                ratatui::layout::Constraint::Min(3),
            ])
            .split(area);

        let header = components::HeaderStatus {
            active_tab: self.mode.tab_index(),
            server_running: self.server_panel_state.phase == ServerPhase::Running,
            server_port: self.server_state.as_ref().map(|s| s.port).or(
                if self.server_panel_state.phase == ServerPhase::Running {
                    Some(self.server_panel_state.port)
                } else {
                    None
                },
            ),
            active_model: self.chat_state.current_model.as_deref(),
        };
        components::render_header_bar(f, chunks[0], &header);

        let content = chunks[1];
        match self.mode {
            AppMode::Chat => {
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
                    &mut self.chat_input,
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
                    &self.hardware,
                    self.config.inference.default_context_size,
                    self.config.inference.default_gpu_layers,
                );
            }
            AppMode::Browser => {
                components::render_model_browser(
                    f,
                    content,
                    &self.browser_state,
                    self.loading_spinner,
                );
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

        // Render modals on top
        if self.api_key_modal.open {
            crate::api_key_modal::render_api_key_modal(f, &self.api_key_modal);
        }
        if self.help_open {
            components::render_help_modal(f, self.mode.tab_index());
        }
    }

    async fn handle_chat_key(&mut self, key: event::KeyEvent) {
        match key.code {
            KeyCode::Enter => {
                // Shift+Enter or Alt+Enter = newline, Enter = send
                if key.modifiers.contains(KeyModifiers::SHIFT)
                    || key.modifiers.contains(KeyModifiers::ALT)
                {
                    self.chat_input.insert_newline();
                } else {
                    self.send_message().await;
                }
            }
            // Ctrl+J = newline (works reliably across all terminals)
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.chat_input.insert_newline();
            }
            // Alt+Up / Alt+Down = prompt history
            KeyCode::Up if key.modifiers.contains(KeyModifiers::ALT) => {
                let current = self.chat_input.lines().join("\n");
                if let Some(prev) = self.chat_state.history_prev(&current) {
                    self.chat_input = new_chat_input();
                    self.chat_input.insert_str(&prev);
                }
            }
            KeyCode::Down if key.modifiers.contains(KeyModifiers::ALT) => {
                if let Some(next) = self.chat_state.history_next() {
                    self.chat_input = new_chat_input();
                    self.chat_input.insert_str(&next);
                }
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
            KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_chat_up(1);
            }
            KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_chat_down(1);
            }
            KeyCode::Up => {
                if self.chat_input.lines().len() <= 1 {
                    self.scroll_chat_up(1);
                } else {
                    self.chat_input.input(tui_textarea::Input::from(key));
                }
            }
            KeyCode::Down => {
                if self.chat_input.lines().len() <= 1 {
                    self.scroll_chat_down(1);
                } else {
                    self.chat_input.input(tui_textarea::Input::from(key));
                }
            }
            KeyCode::PageUp => {
                self.scroll_chat_up(20);
            }
            KeyCode::PageDown => {
                self.scroll_chat_down(20);
            }
            KeyCode::Home => {
                self.chat_state.auto_scroll = false;
                self.chat_state.scroll = 0;
            }
            KeyCode::End => {
                self.chat_state.auto_scroll = true;
                self.chat_state.scroll = 0;
            }
            KeyCode::Esc if self.chat_state.is_generating => {
                self.cancel_generation();
            }
            _ => {
                self.chat_input.input(tui_textarea::Input::from(key));
                self.chat_state.auto_scroll = true;
            }
        }
    }

    async fn handle_model_list_key(&mut self, key: event::KeyEvent) {
        // Delete confirmation modal
        if let Some((name, _size)) = self.model_list_state.confirm_delete.clone() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.model_list_state.confirm_delete = None;
                    self.model_list_state
                        .set_status(format!("Deleting '{}'…", name));
                    let models_dir = self.config.paths.models_dir.clone();
                    self.model_delete_task = Some(tokio::task::spawn_blocking(move || {
                        let registry = ModelRegistry::new(models_dir);
                        let res = registry.remove_model(&name).map_err(|e| e.to_string());
                        (name, res)
                    }));
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.model_list_state.confirm_delete = None;
                }
                _ => {}
            }
            return;
        }

        // Live filter input mode
        if self.model_list_state.filtering {
            match key.code {
                KeyCode::Esc => {
                    self.model_list_state.filtering = false;
                    self.model_list_state.filter.clear();
                    self.model_list_state.clamp_selection();
                }
                KeyCode::Enter => {
                    self.model_list_state.filtering = false;
                }
                KeyCode::Backspace => {
                    self.model_list_state.filter.pop();
                    self.model_list_state.clamp_selection();
                }
                KeyCode::Char(c) => {
                    self.model_list_state.filter.push(c);
                    self.model_list_state.clamp_selection();
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.model_list_state.next();
                self.model_list_state.clear_status();
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.model_list_state.previous();
                self.model_list_state.clear_status();
            }
            KeyCode::Char('/') => {
                self.model_list_state.filtering = true;
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
                    self.model_list_state.confirm_delete = Some((
                        model.name.clone(),
                        theme::format_bytes_binary(model.file_size_bytes),
                    ));
                }
            }
            KeyCode::Char('u') => {
                if self.backend.is_some() {
                    let name = self.chat_state.current_model.clone().unwrap_or_default();
                    let backend_name = self.chat_state.current_backend.clone().unwrap_or_default();
                    self.backend = None;
                    self.chat_state.current_model = None;
                    self.chat_state.current_backend = None;
                    self.chat_stream_rx = None;
                    self.chat_cancel = None;
                    self.log(&format!(
                        "Model '{}' [{}] unloaded from memory",
                        name, backend_name
                    ));
                    self.model_list_state
                        .set_status(format!("Unloaded '{}' [{}]", name, backend_name));
                    self.chat_state.add_message(
                        "system",
                        &format!("Model '{}' [{}] unloaded from memory.", name, backend_name),
                    );
                } else if self.server_state.is_some()
                    && !self.server_panel_state.loaded_models.is_empty()
                {
                    // Unload from detached server as well
                    self.unload_model_action().await;
                    if let Some(msg) = self.server_panel_state.status_message.clone() {
                        if self.server_panel_state.status_is_error {
                            self.model_list_state.set_error(msg);
                        } else {
                            self.model_list_state.set_status(msg);
                        }
                    }
                    if self.server_panel_state.loaded_models.is_empty() {
                        self.chat_state.current_model = None;
                        self.chat_state.current_backend = None;
                    }
                } else {
                    self.model_list_state
                        .set_error("No model is currently loaded");
                }
            }
            KeyCode::Esc => {
                if !self.model_list_state.filter.is_empty() {
                    self.model_list_state.filter.clear();
                    self.model_list_state.clamp_selection();
                } else {
                    self.mode = AppMode::Chat;
                }
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
                KeyCode::Enter => match self.settings_state.apply_edit() {
                    Err(e) => self.settings_state.set_error(e),
                    Ok(()) => {
                        self.sync_config_from_settings();
                        match self.save_config_async(self.config.clone()).await {
                            Ok(()) => self.settings_state.set_status("Saved!"),
                            Err(e) => {
                                self.settings_state
                                    .set_error(format!("Failed to save: {}", e));
                            }
                        }
                    }
                },
                KeyCode::Backspace => {
                    self.settings_state.edit_buffer.pop();
                }
                KeyCode::Char(c) => {
                    self.settings_state.edit_buffer.push(c);
                }
                _ => {}
            }
        } else {
            match key.code {
                KeyCode::Down | KeyCode::Char('j') => {
                    self.settings_state.next();
                    self.settings_state.clear_status();
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.settings_state.previous();
                    self.settings_state.clear_status();
                }
                KeyCode::Enter | KeyCode::Char(' ') => {
                    if let Some(summary) = self.settings_state.toggle_or_cycle_current() {
                        self.sync_config_from_settings();
                        match self.save_config_async(self.config.clone()).await {
                            Ok(()) => {
                                self.settings_state
                                    .set_status(format!("{} — saved", summary));
                                self.log(&summary);
                            }
                            Err(e) => {
                                self.settings_state
                                    .set_error(format!("Failed to save: {}", e));
                            }
                        }
                    } else if key.code == KeyCode::Enter {
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
        if self.logs_state.searching {
            match key.code {
                KeyCode::Esc => {
                    self.logs_state.searching = false;
                    self.logs_state.search_query.clear();
                }
                KeyCode::Enter => {
                    self.logs_state.searching = false;
                }
                KeyCode::Backspace => {
                    self.logs_state.search_query.pop();
                }
                KeyCode::Char(c) => {
                    self.logs_state.search_query.push(c);
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.logs_state.scroll_up(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.logs_state.scroll_down(1);
            }
            KeyCode::PageUp => {
                self.logs_state.scroll_up(15);
            }
            KeyCode::PageDown => {
                self.logs_state.scroll_down(15);
            }
            KeyCode::End => {
                self.logs_state.scroll_to_bottom();
            }
            KeyCode::Char('c') => {
                self.logs_state.clear();
            }
            KeyCode::Char('a') => {
                self.logs_state.level_filter = LogLevelFilter::All;
                self.logs_state.scroll_to_bottom();
            }
            KeyCode::Char('e') => {
                self.logs_state.level_filter = LogLevelFilter::Error;
                self.logs_state.scroll_to_bottom();
            }
            KeyCode::Char('w') => {
                self.logs_state.level_filter = LogLevelFilter::Warn;
                self.logs_state.scroll_to_bottom();
            }
            KeyCode::Char('i') => {
                self.logs_state.level_filter = LogLevelFilter::Info;
                self.logs_state.scroll_to_bottom();
            }
            KeyCode::Char('d') => {
                self.logs_state.level_filter = LogLevelFilter::Debug;
                self.logs_state.scroll_to_bottom();
            }
            KeyCode::Char('/') => {
                self.logs_state.searching = true;
            }
            KeyCode::Char('z') | KeyCode::Char('v') => {
                self.logs_state.wrap_lines = !self.logs_state.wrap_lines;
            }
            KeyCode::Esc => {
                if !self.logs_state.search_query.is_empty() {
                    self.logs_state.search_query.clear();
                } else {
                    self.mode = AppMode::Chat;
                }
            }
            _ => {}
        }
    }

    async fn handle_browser_key(&mut self, key: event::KeyEvent) {
        match &self.browser_state.phase {
            BrowserPhase::Search => match key.code {
                KeyCode::Enter => {
                    let query = self.browser_state.search_input.trim().to_string();
                    if !query.is_empty() && !self.browser_state.is_busy {
                        self.start_browser_search(query);
                    }
                }
                KeyCode::Backspace => {
                    self.browser_state.search_input.pop();
                }
                KeyCode::Char('g') | KeyCode::Char('G')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.browser_state.gguf_only = !self.browser_state.gguf_only;
                }
                KeyCode::Esc => {
                    if let Some(h) = self.browser_search_task.take() {
                        h.abort();
                    }
                    self.browser_state.is_busy = false;
                    self.browser_state.busy_label = None;
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
                    if !self.browser_state.is_busy {
                        if let Some(result) = self.browser_state.selected_result() {
                            let repo_id = result.id.clone();
                            self.start_browser_list_files(repo_id);
                        }
                    }
                }
                KeyCode::Esc | KeyCode::Char('/') => {
                    if let Some(h) = self.browser_files_task.take() {
                        h.abort();
                    }
                    self.browser_state.is_busy = false;
                    self.browser_state.busy_label = None;
                    self.browser_state.back_to_search_edit();
                }
                KeyCode::Char('r') | KeyCode::Char('R') => {
                    if let Some(h) = self.browser_files_task.take() {
                        h.abort();
                    }
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
                    if let Some((filename, _, sha256)) = self
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
                        self.start_download(&repo_id, &filename, sha256);
                    }
                }
                KeyCode::Esc => {
                    self.browser_state.phase = BrowserPhase::Results;
                }
                _ => {}
            },
            BrowserPhase::Downloading => {
                if key.code == KeyCode::Esc {
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

    fn start_browser_search(&mut self, query: String) {
        if let Some(h) = self.browser_search_task.take() {
            h.abort();
        }
        self.browser_state.is_busy = true;
        self.browser_state.busy_label = Some(format!("Searching HuggingFace for '{}'…", query));
        self.browser_state.status_message = None;
        self.browser_state.status_is_error = false;

        let token = self.config.huggingface.token.clone();
        let gguf_only = self.browser_state.gguf_only;
        self.browser_search_task = Some(tokio::spawn(async move {
            let client = athenas_hub::HuggingFaceClient::new(token);
            let filters = athenas_hub::ModelSearchFilters {
                pipeline_tag: None,
                library_name: None,
                gguf_only,
                safetensors_only: false,
            };
            client
                .search_models(&query, &filters)
                .await
                .map_err(|e| e.to_string())
        }));
    }

    async fn poll_browser_search(&mut self) {
        let Some(task) = self.browser_search_task.take() else {
            return;
        };
        if !task.is_finished() {
            self.browser_search_task = Some(task);
            return;
        }
        self.browser_state.is_busy = false;
        self.browser_state.busy_label = None;
        match task.await {
            Ok(Ok(results)) => {
                self.browser_state.search_results = results;
                self.browser_state.results_selected = 0;
                self.browser_state.phase = BrowserPhase::Results;
                self.browser_state.status_message = None;
                self.browser_state.status_is_error = false;
            }
            Ok(Err(e)) => {
                self.browser_state.status_message = Some(format!("Search failed: {}", e));
                self.browser_state.status_is_error = true;
            }
            Err(e) => {
                self.browser_state.status_message = Some(format!("Search task failed: {}", e));
                self.browser_state.status_is_error = true;
            }
        }
    }

    fn start_browser_list_files(&mut self, repo_id: String) {
        if let Some(h) = self.browser_files_task.take() {
            h.abort();
        }
        self.browser_state.is_busy = true;
        self.browser_state.busy_label = Some(format!("Fetching files from {}…", repo_id));
        self.browser_state.status_message = None;
        self.browser_state.status_is_error = false;

        let token = self.config.huggingface.token.clone();
        self.browser_files_task = Some(tokio::spawn(async move {
            let client = athenas_hub::HuggingFaceClient::new(token);
            let files = client
                .get_model_files_recursive(&repo_id, "main")
                .await
                .map_err(|e| e.to_string())?;

            let mut options: Vec<(String, Option<u64>, Option<String>)> = files
                .iter()
                .filter(|f| f.path.ends_with(".gguf"))
                .map(|f| {
                    (
                        f.path.clone(),
                        f.size.or(f.lfs.as_ref().and_then(|l| l.size)),
                        f.lfs.as_ref().and_then(|l| l.sha256.clone()),
                    )
                })
                .collect();

            let mut onnx_groups: Vec<(String, u64)> = Vec::new();
            for f in files.iter().filter(|f| f.path.ends_with(".onnx")) {
                let dir = f
                    .path
                    .rsplit_once('/')
                    .map(|(d, _)| d.to_string())
                    .unwrap_or_default();
                let size: u64 = files
                    .iter()
                    .filter(|g| {
                        g.path
                            .rsplit_once('/')
                            .map(|(d, _)| d.to_string())
                            .unwrap_or_default()
                            == dir
                    })
                    .map(|g| g.size.or(g.lfs.as_ref().and_then(|l| l.size)).unwrap_or(0))
                    .sum();
                if !onnx_groups.iter().any(|(d, _)| d == &dir) {
                    onnx_groups.push((dir, size));
                }
            }
            for (dir, size) in onnx_groups {
                options.push((format!("onnx-dir:{}", dir), Some(size), None));
            }
            Ok(options)
        }));
    }

    async fn poll_browser_files(&mut self) {
        let Some(task) = self.browser_files_task.take() else {
            return;
        };
        if !task.is_finished() {
            self.browser_files_task = Some(task);
            return;
        }
        self.browser_state.is_busy = false;
        self.browser_state.busy_label = None;
        match task.await {
            Ok(Ok(options)) => {
                if options.is_empty() {
                    self.browser_state.status_message = Some(
                        "No GGUF or ONNX files found in this repo. Try a quantized version."
                            .to_string(),
                    );
                    self.browser_state.status_is_error = true;
                } else {
                    self.browser_state.file_options = options;
                    self.browser_state.file_selected = 0;
                    self.browser_state.phase = BrowserPhase::SelectFile;
                    self.browser_state.status_message = None;
                    self.browser_state.status_is_error = false;
                }
            }
            Ok(Err(e)) => {
                self.browser_state.status_message = Some(format!("Failed to list files: {}", e));
                self.browser_state.status_is_error = true;
            }
            Err(e) => {
                self.browser_state.status_message = Some(format!("File list task failed: {}", e));
                self.browser_state.status_is_error = true;
            }
        }
    }

    fn start_download(&mut self, repo_id: &str, filename: &str, expected_sha256: Option<String>) {
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
            let result = if let Some(variant) = filename_owned.strip_prefix("onnx-dir:") {
                downloader
                    .download_model_dir(&repo_id_owned, "main", variant, Some(tx))
                    .await
            } else {
                downloader
                    .download_model_verify(
                        &repo_id_owned,
                        &filename_owned,
                        "main",
                        Some(tx),
                        expected_sha256,
                    )
                    .await
            };

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
                            .download_model_verify(
                                &repo_id_owned,
                                &mmproj.path,
                                "main",
                                Some(tx_clone.clone()),
                                mmproj.lfs.as_ref().and_then(|l| l.sha256.clone()),
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
                        self.spawn_refresh_models();
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
        // Get text from textarea (join multi-line with newlines)
        let text = self.chat_input.lines().join("\n").trim().to_string();
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

        self.chat_state.push_history(text.clone());
        self.chat_state.add_message("user", &text);
        self.chat_input = new_chat_input();
        self.chat_state.is_generating = true;
        self.chat_state.generation_start = Some(std::time::Instant::now());

        // Build chat request from current messages
        let mut messages: Vec<ChatMessage> = Vec::new();

        if !self.chat_state.system_prompt.is_empty() {
            messages.push(ChatMessage {
                role: Role::System,
                content: MessageContent::Text(self.chat_state.system_prompt.clone()),
            });
        }

        messages.extend(
            self.chat_state
                .messages
                .iter()
                .filter(|m| m.role != "system" && m.role != "error")
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
                }),
        );

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

        let backend_ref: Option<Box<dyn Backend>> = if let Some(ref b) = self.backend {
            Some(b.boxed_clone())
        } else if let Some(mgr) = &self.shared_model_manager {
            let m = mgr.lock().await;
            m.get(None).map(|b| b.boxed_clone())
        } else if let Some(ref state) = self.server_state {
            let api_key = self.server_panel_state.auth_bearer();
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
            self.chat_state.is_generating = false;
            self.chat_state.generation_start = None;
            self.chat_state
                .add_error_message("No backend available for inference.");
            return;
        };

        if !self.config.inference.streaming_enabled {
            let (tx, rx) = tokio::sync::mpsc::channel::<ChatStreamEvent>(100);
            let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
            cancel_rx.borrow_and_update();
            tokio::spawn(async move {
                tokio::select! {
                    result = backend.chat(req) => {
                        match result {
                            Ok(resp) => {
                                let _ = tx
                                    .send(ChatStreamEvent::Chunk(StreamChunk {
                                        text: resp.message.content.as_text(),
                                        done: false,
                                        is_reasoning: false,
                                        stats: None,
                                    }))
                                    .await;
                                let _ = tx
                                    .send(ChatStreamEvent::Chunk(StreamChunk {
                                        text: String::new(),
                                        done: true,
                                        is_reasoning: false,
                                        stats: Some(resp.stats),
                                    }))
                                    .await;
                            }
                            Err(e) => {
                                tracing::error!("Chat error: {}", e);
                                let _ = tx.send(ChatStreamEvent::Error(e.to_string())).await;
                            }
                        }
                    }
                    _ = cancel_rx.changed() => {
                        tracing::info!("Chat request cancelled by user");
                    }
                }
            });
            self.chat_stream_rx = Some(rx);
            self.chat_cancel = Some(cancel_tx);
            return;
        }

        // Start streaming in background — bridge StreamChunk into ChatStreamEvent so errors surface in Chat
        let (event_tx, event_rx) = tokio::sync::mpsc::channel::<ChatStreamEvent>(100);
        let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<StreamChunk>(100);
        let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);
        cancel_rx.borrow_and_update();

        let forward_tx = event_tx.clone();
        tokio::spawn(async move {
            while let Some(chunk) = chunk_rx.recv().await {
                if forward_tx
                    .send(ChatStreamEvent::Chunk(chunk))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        tokio::spawn(async move {
            tokio::select! {
                result = backend.chat_stream(req, chunk_tx) => {
                    if let Err(e) = result {
                        tracing::error!("Chat stream error: {}", e);
                        let _ = event_tx.send(ChatStreamEvent::Error(e.to_string())).await;
                    }
                }
                _ = cancel_rx.changed() => {
                    tracing::info!("Chat stream cancelled by user");
                }
            }
        });

        self.chat_stream_rx = Some(event_rx);
        self.chat_cancel = Some(cancel_tx);
    }

    /// Cancel the ongoing chat generation.
    fn cancel_generation(&mut self) {
        if let Some(cancel_tx) = &self.chat_cancel {
            let _ = cancel_tx.send(true);
        }
        self.chat_cancel = None;

        let had_content = !self.chat_state.streaming_text.is_empty()
            || !self.chat_state.streaming_reasoning.is_empty();
        self.chat_state.finalize_streaming();

        if !had_content {
            self.chat_state
                .add_message("system", "Generation cancelled.");
        } else if let Some(last) = self.chat_state.messages.last_mut() {
            if last.role == "assistant" {
                last.content
                    .push_str("\n\n_⚠ Generation cancelled by user._");
                let last_idx = self.chat_state.messages.len().saturating_sub(1);
                self.chat_state.invalidate_md_cache(last_idx);
            }
        }

        self.chat_stream_rx = None;
    }

    async fn poll_chat_stream(&mut self) {
        if !self.chat_state.is_generating {
            return;
        }

        if let Some(start) = self.chat_state.generation_start {
            let elapsed = start.elapsed().as_secs();
            let has_output = !self.chat_state.streaming_text.is_empty()
                || !self.chat_state.streaming_reasoning.is_empty();
            if elapsed > 120 && !has_output {
                self.chat_state.add_error_message(
                    "Request timed out (120s with no output). The model may not be responding. \
                     Try a smaller model, reduce context size, or disable reasoning in Settings.",
                );
                self.chat_state.finalize_streaming();
                self.chat_stream_rx = None;
                self.chat_cancel = None;
                return;
            }
        }

        if let Some(rx) = &mut self.chat_stream_rx {
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    ChatStreamEvent::Error(err) => {
                        self.chat_state.finalize_streaming();
                        self.chat_state
                            .add_error_message(&format!("Generation failed: {}", err));
                        self.chat_stream_rx = None;
                        self.chat_cancel = None;
                        return;
                    }
                    ChatStreamEvent::Chunk(chunk) => {
                        if chunk.done {
                            if let Some(stats) = chunk.stats {
                                self.chat_state.tokens_per_second = Some(stats.tokens_per_second);
                            }
                            self.chat_state.finalize_streaming();
                            self.chat_stream_rx = None;
                            self.chat_cancel = None;
                            return;
                        } else {
                            if chunk.is_reasoning {
                                self.chat_state.append_reasoning(&chunk.text);
                            } else {
                                self.chat_state.append_streaming(&chunk.text);
                            }
                            if let Some(stats) = &chunk.stats {
                                self.chat_state.tokens_per_second = Some(stats.tokens_per_second);
                            }
                        }
                    }
                }
            }

            if rx.is_closed() {
                if self.chat_state.is_generating {
                    self.chat_state.finalize_streaming();
                }
                self.chat_stream_rx = None;
                self.chat_cancel = None;
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
                self.switch_to_mode(AppMode::ModelList);
            }
            "/browser" => {
                self.mode = AppMode::Browser;
            }
            "/server" => {
                self.switch_to_mode(AppMode::Server);
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
                     /system <prompt> — set system prompt for the model\n\
                     /system — show current system prompt\n\
                     /system clear — remove system prompt\n\
                     Alt+Enter / Ctrl+J / Shift+Enter — insert newline\n\
                     Alt+↑ / Alt+↓ — recall prompt history\n\
                     Tab — expand/collapse Thinking | Esc — cancel generation\n\
                     F1–F6 or Alt+1–6: Switch tabs | Ctrl+C: Quit",
                );
            }
            "/system" => {
                if parts.len() < 2 || parts[1].trim().is_empty() {
                    if self.chat_state.system_prompt.is_empty() {
                        self.chat_state.add_message(
                            "system",
                            "No system prompt set. Use /system <prompt> to set one.",
                        );
                    } else {
                        self.chat_state.add_message(
                            "system",
                            &format!("Current system prompt:\n{}", self.chat_state.system_prompt),
                        );
                    }
                } else if parts[1].trim() == "clear" {
                    self.chat_state.system_prompt.clear();
                    self.chat_state
                        .add_message("system", "System prompt cleared.");
                } else {
                    self.chat_state.system_prompt = parts[1].trim().to_string();
                    self.chat_state.add_message(
                        "system",
                        &format!("System prompt set to:\n{}", self.chat_state.system_prompt),
                    );
                }
            }
            "/quit" => {
                self.chat_state.add_message("system", "Use Ctrl+C to quit");
            }
            _ => {
                self.chat_state
                    .add_message("system", &format!("Unknown command: {}", parts[0]));
            }
        }
        self.chat_input = new_chat_input();
    }

    async fn load_model(&mut self, path: &str) {
        if self.is_loading_model {
            self.chat_state
                .add_message("system", "Already loading a model, please wait...");
            return;
        }

        // Reject multimodal projector files — they are not standalone models
        let filename_lower = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        if filename_lower.contains("mmproj") {
            self.chat_state.add_message(
                "system",
                "Error: This is a multimodal projector file (-mmproj.gguf), \
                 not a standalone model.\n\
                 Select the main model file (e.g. Q4_K_M.gguf, F16.gguf) instead.\n\
                 The mmproj file is used automatically for vision models.",
            );
            return;
        }

        // === Resource protections ===

        // Skip auto-capping if user disabled it
        let auto_limits = self.config.inference.auto_resource_limits;

        // 1. Check model file size vs available RAM (non-blocking metadata read)
        let path_owned = path.to_string();
        let model_size_mb = tokio::task::spawn_blocking(move || {
            std::fs::metadata(&path_owned)
                .map(|m| m.len() / (1024 * 1024))
                .unwrap_or(0)
        })
        .await
        .unwrap_or(0);

        let avail_mb = self.hardware.memory_available_mb;
        let total_mb = self.hardware.memory_total_mb;

        // Shared heuristic: file size + context/KV-cache overhead, split
        // between host RAM and VRAM according to the gpu_layers setting.
        let est = athenas_core::estimate_model_memory(
            model_size_mb,
            self.config.inference.default_context_size,
            self.config.inference.default_gpu_layers,
            &self.hardware,
        );

        if auto_limits && !est.fits(&self.hardware) {
            let (resource, needed, avail) = est
                .shortfall(&self.hardware)
                .unwrap_or(("RAM", est.ram_mb, avail_mb));
            self.chat_state.add_message(
                "system",
                &format!(
                    "⚠ Not enough {} to load this model safely.\n\
                     Model: {}MB, estimated need: {}MB, available: {}MB\n\
                     Try a smaller model, smaller context size, more GPU layers, \
                     or close other applications.",
                    resource, model_size_mb, needed, avail
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

        // 3. Cap context size based on available memory (overflow-safe u64 math)
        let mut context_size = self.config.inference.default_context_size;
        if auto_limits {
            let max_ctx_by_mem: u32 = if total_mb > 0 {
                // Reserve model size + ram_reserve_mb, allow up to 40% of remaining for KV cache
                let reserved = model_size_mb.saturating_add(self.config.inference.ram_reserve_mb);
                let usable = total_mb.saturating_sub(reserved);
                // ~64 MB per 1024 context tokens
                let ctx_mb = (usable.saturating_mul(4)) / 10;
                let ctx_tokens = (ctx_mb / 64).saturating_mul(1024);
                u32::try_from(ctx_tokens).unwrap_or(u32::MAX)
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
            draft_model_path: None,
            draft_max_tokens: 16,
            draft_min_ctx: 512,
            auto_install_deps: self.config.inference.auto_install_deps,
        };

        let task = tokio::spawn(async move {
            let mut backend =
                BackendFactory::create_for_model(backend_type, &hardware, &load_config.model_path)?;
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
                KeyCode::Enter => match self.server_panel_state.save_edit() {
                    Err(e) => self.server_panel_state.set_error(e),
                    Ok(()) => {
                        let config = self.sync_config_from_server_panel();
                        if let Err(e) = self.save_config_async(config).await {
                            self.server_panel_state
                                .set_error(format!("Failed to save config: {}", e));
                        } else if self.server_panel_state.phase == ServerPhase::Running {
                            self.server_panel_state
                                .set_status("Saved — applies on next server start");
                        } else {
                            self.server_panel_state.set_status("Saved");
                        }
                    }
                },
                KeyCode::Backspace => {
                    self.server_panel_state.edit_buffer.pop();
                }
                KeyCode::Char(c) => {
                    self.server_panel_state.edit_buffer.push(c);
                }
                _ => {}
            }
        } else {
            let field = self.server_panel_state.current_field().clone();
            match key.code {
                // Sub-tab switching with [ and ]
                KeyCode::Char('[') => {
                    self.server_panel_state.prev_subtab();
                    self.server_panel_state.clear_status();
                }
                KeyCode::Char(']') => {
                    self.server_panel_state.next_subtab();
                    self.server_panel_state.clear_status();
                }
                // Pinned quick actions: s (start/stop), l (load additional), u (unload), k (API keys)
                KeyCode::Char('s') => {
                    if self.server_panel_state.phase == ServerPhase::Running
                        || self.server_panel_state.phase == ServerPhase::LoadingModel
                    {
                        self.stop_server().await;
                    } else {
                        self.start_server().await;
                    }
                }
                KeyCode::Char('l') => {
                    self.load_additional_model().await;
                }
                KeyCode::Char('u') => {
                    self.unload_model_action().await;
                }
                KeyCode::Char('k') => {
                    self.api_key_modal.open();
                    if self.server_state.is_none() {
                        self.api_key_modal
                            .set_error("Server not running — start it to manage keys".to_string());
                    } else {
                        self.refresh_api_keys().await;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.server_panel_state.next();
                    self.server_panel_state.clear_status();
                }
                KeyCode::Up => {
                    self.server_panel_state.previous();
                    self.server_panel_state.clear_status();
                }
                KeyCode::PageDown => {
                    self.server_panel_state.jump_forward(10);
                    self.server_panel_state.clear_status();
                }
                KeyCode::PageUp => {
                    self.server_panel_state.jump_back(10);
                    self.server_panel_state.clear_status();
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
                    _ => {
                        self.server_panel_state.prev_subtab();
                    }
                },
                KeyCode::Right => match field {
                    ConfigField::ModelSelection => {
                        self.server_panel_state.select_model_next();
                    }
                    ConfigField::UnloadModel => {
                        self.server_panel_state.unload_select_next();
                    }
                    ConfigField::SetDefaultModel => {
                        self.server_panel_state.default_select_next();
                    }
                    _ => {
                        self.server_panel_state.next_subtab();
                    }
                },
                KeyCode::Enter | KeyCode::Char(' ') => {
                    if field.is_toggle() {
                        self.server_panel_state.toggle();
                        let config = self.sync_config_from_server_panel();
                        if let Err(e) = self.save_config_async(config).await {
                            self.server_panel_state
                                .set_error(format!("Failed to save config: {}", e));
                        } else if self.server_panel_state.phase == ServerPhase::Running {
                            self.server_panel_state
                                .set_status("Toggled — applies on next server start");
                        } else {
                            self.server_panel_state.set_status("Toggled — saved");
                        }
                    } else if field == ConfigField::GpuRuntime {
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
                        let config = self.sync_config_from_server_panel();
                        if let Err(e) = self.save_config_async(config).await {
                            self.server_panel_state
                                .set_error(format!("Failed to save config: {}", e));
                        } else {
                            self.server_panel_state
                                .set_status(format!("GPU runtime: {} — saved", next));
                        }
                    } else if key.code == KeyCode::Enter {
                        if field.is_editable() {
                            self.server_panel_state.start_edit();
                        } else if field == ConfigField::StartServer {
                            self.start_server().await;
                        } else if field == ConfigField::StopServer {
                            self.stop_server().await;
                        } else if field == ConfigField::LoadAdditionalModel {
                            self.load_additional_model().await;
                        } else if field == ConfigField::UnloadModel {
                            self.unload_model_action().await;
                        } else if field == ConfigField::SetDefaultModel {
                            self.set_default_model_action().await;
                        } else if field == ConfigField::ManageApiKeys {
                            self.api_key_modal.open();
                            if self.server_state.is_none() {
                                self.api_key_modal.set_error(
                                    "Server not running — start it to manage keys".to_string(),
                                );
                            } else {
                                self.refresh_api_keys().await;
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
    }

    async fn start_server(&mut self) {
        if self.server_panel_state.phase == ServerPhase::Running {
            self.server_panel_state
                .set_status("Server is already running");
            return;
        }

        // Don't start if already loading — check BOTH the legacy in-process
        // task AND the detached health task. The detached path uses
        // server_health_task, not server_start_task.
        if self.server_start_task.is_some() || self.server_health_task.is_some() {
            self.server_panel_state
                .set_status("Server is already starting, please wait...");
            return;
        }

        // If phase is Error or LoadingModel from a previous attempt, make
        // sure any leftover process is killed before starting a new one.
        if self.server_panel_state.phase == ServerPhase::Error
            || self.server_panel_state.phase == ServerPhase::LoadingModel
        {
            if let Some(ref state) = self.server_state {
                let pid = state.pid;
                // Kill in background to avoid blocking the TUI
                let stale_pid = pid;
                let _ = tokio::task::spawn_blocking(move || {
                    let _ = server_manager::stop_by_pid(stale_pid);
                })
                .await;
            }
            self.server_state = None;
        }

        let model_path = match self.server_panel_state.selected_model_path() {
            Some(p) => p,
            None => {
                self.server_panel_state
                    .set_error("No model selected. Use Left/Right to pick a model.");
                return;
            }
        };

        // Reject multimodal projector files
        let model_name_lower = std::path::Path::new(&model_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        if model_name_lower.contains("mmproj") {
            self.server_panel_state.set_error(
                "Error: This is a multimodal projector (-mmproj), not a standalone model. \
                 Select the main model file (e.g. Q4_K_M.gguf) instead.",
            );
            return;
        }

        // Check if the port is already in use — fail early, BEFORE any
        // state/config changes, instead of spawning a doomed process.
        let host = self.server_panel_state.host.clone();
        let port = self.server_panel_state.port;
        if std::net::TcpListener::bind((host.as_str(), port)).is_err() {
            self.server_panel_state.phase = ServerPhase::Error;
            self.server_panel_state.set_error(format!(
                "Port {} is already in use. Stop the existing server or change the port.",
                port
            ));
            return;
        }

        // Pre-flight memory check — refuse to start when the model clearly
        // does not fit in the available RAM/VRAM. Uses fresh memory data.
        {
            let (total, avail) = tokio::task::spawn_blocking(athenas_core::detect_memory_mb)
                .await
                .unwrap_or((0, 0));
            if total > 0 {
                self.server_panel_state.hardware.memory_total_mb = total;
                self.hardware.memory_total_mb = total;
            }
            if avail > 0 {
                self.server_panel_state.hardware.memory_available_mb = avail;
                self.hardware.memory_available_mb = avail;
            }
        }
        if self.server_panel_state.auto_resource_limits {
            if let Some(est) = self.server_panel_state.estimate_selected_model_load() {
                if !est.fits {
                    // For APUs with full GPU offload, the bottleneck is system
                    // RAM (unified memory), not dedicated VRAM
                    let (need, avail) = if est.full_gpu_offload
                        && est.vram_free_mb.unwrap_or(0) < est.vram_mb.unwrap_or(0)
                    {
                        // Show the larger of the two needs (vram vs ram)
                        // and the relevant available memory
                        let vram_need = est.vram_mb.unwrap_or(0);
                        let ram_need = est.ram_mb;
                        if vram_need > ram_need {
                            (
                                vram_need,
                                est.ram_available_mb.max(est.vram_free_mb.unwrap_or(0)),
                            )
                        } else {
                            (ram_need, est.ram_available_mb)
                        }
                    } else if est.full_gpu_offload {
                        (est.vram_mb.unwrap_or(0), est.vram_free_mb.unwrap_or(0))
                    } else {
                        (est.ram_mb, est.ram_available_mb)
                    };
                    self.server_panel_state.phase = ServerPhase::Error;
                    self.server_panel_state.set_error(format!(
                        "Not enough memory: estimated need ~{:.1} GB, only {:.1} GB available. \
                         Use a smaller model/quant, lower Context Size, or turn off \
                         Auto Resource Limits to override.",
                        need as f64 / 1024.0,
                        avail as f64 / 1024.0,
                    ));
                    return;
                }
            }
        }

        // Get model name for display
        let model_name = self
            .server_panel_state
            .models
            .get(self.server_panel_state.model_selected)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| model_path.clone());

        self.server_panel_state.phase = ServerPhase::LoadingModel;
        self.server_panel_state
            .set_status(format!("Starting server with model: {}...", model_name));

        // Save server config so it persists across restarts (via spawn_blocking)
        let server_config = self.sync_config_from_server_panel();
        if let Err(e) = self.save_config_async(server_config).await {
            tracing::warn!("Failed to save server config: {}", e);
        }

        let backend_str = match self.server_panel_state.backend {
            athenas_core::BackendType::Auto => "auto",
            athenas_core::BackendType::LlamaCpp => "llama.cpp",
            athenas_core::BackendType::Vllm => "vllm",
            athenas_core::BackendType::Onnx => "onnx",
        };
        let gpu_layers = self.server_panel_state.gpu_layers;
        let gpu_runtime = self.server_panel_state.gpu_runtime.to_string();
        let gpu_device = self.server_panel_state.gpu_device;
        let context_size = self.server_panel_state.context_size;
        let threads = self.server_panel_state.threads;
        let batch_size = self.server_panel_state.batch_size;
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
            Some(threads),
            Some(batch_size),
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
                self.server_panel_state
                    .set_status(format!("Loading model (PID: {})...", health_pid));
                self.server_panel_state.loaded_model_name = Some(model_name.clone());
                self.server_panel_state.loaded_backend_name = Some(backend_str.to_string());
                self.server_state = Some(state);

                // Spawn a background task to poll the health endpoint
                // Use 127.0.0.1 for the health check even if the server
                // binds to 0.0.0.0 — on Windows, connecting to 0.0.0.0
                // doesn't work like it does on Linux.
                let health_host = if host == "0.0.0.0" || host == "::" {
                    "127.0.0.1".to_string()
                } else {
                    host.clone()
                };
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
                self.server_panel_state
                    .set_error(format!("Failed to start server: {}", e));
            }
        }
    }

    /// Get the connectable host for the server.
    /// If the server binds to 0.0.0.0 or ::, use 127.0.0.1 for local
    /// connections — on Windows, connecting to 0.0.0.0 doesn't work.
    fn connect_host(&self) -> String {
        if let Some(ref state) = self.server_state {
            if state.host == "0.0.0.0" || state.host == "::" {
                "127.0.0.1".to_string()
            } else {
                state.host.clone()
            }
        } else {
            "127.0.0.1".to_string()
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
                self.server_panel_state
                    .set_status(format!("Server running (PID: {})", state.pid));
                self.chat_state.current_model = Some(state.model);
                self.chat_state.current_backend = Some(state.backend);
                // Fetch loaded models list from the detached server
                self.refresh_remote_loaded_models().await;
                // Auto-load API keys so auth works for subsequent requests
                self.refresh_api_keys().await;
            }
            Ok(None) => {
                // No running server found
                if self.server_panel_state.phase == ServerPhase::LoadingModel {
                    // We were waiting for a server we started — it timed out
                    self.server_panel_state.phase = ServerPhase::Error;
                    self.server_panel_state.set_error(
                        "Server failed to start. Check ~/.athenas/server.log for details.",
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
                self.server_panel_state.clear_status();

                self.server_handle = Some(server_handle);
                self.log(&format!("Server started on {}:{}", host, port));

                // Auto-load API keys so auth works for subsequent requests
                self.refresh_api_keys().await;

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
                self.server_panel_state
                    .set_error(format!("Failed to start server: {}", e));
                self.log(&format!("Failed to start server: {}", e));
            }
            Err(e) => {
                self.server_panel_state.phase = ServerPhase::Error;
                self.server_panel_state
                    .set_error(format!("Server start task crashed: {}", e));
                self.log(&format!("Server start task crashed: {}", e));
            }
        }
    }

    async fn stop_server(&mut self) {
        // Always try to kill the detached process, regardless of phase.
        // This handles Running, LoadingModel, AND Error states.
        let mut had_process = false;

        // Cancel any health check task
        if let Some(task) = self.server_health_task.take() {
            task.abort();
        }
        // Cancel any legacy in-process start task
        if let Some(task) = self.server_start_task.take() {
            task.abort();
        }
        // Abort legacy in-process server handle
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }

        // Kill the detached process by PID (in background to avoid
        // blocking the TUI — stop_by_pid does std::thread::sleep)
        if let Some(ref state) = self.server_state {
            had_process = true;
            let pid = state.pid;
            let result =
                tokio::task::spawn_blocking(move || server_manager::stop_by_pid(pid)).await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if !e.contains("not running") {
                        self.server_panel_state
                            .set_error(format!("Error stopping server: {}", e));
                    }
                }
                Err(e) => {
                    tracing::warn!("stop_by_pid task failed: {}", e);
                }
            }
        }

        if !had_process
            && self.server_panel_state.phase != ServerPhase::Running
            && self.server_panel_state.phase != ServerPhase::LoadingModel
            && self.server_panel_state.phase != ServerPhase::Error
        {
            self.server_panel_state.set_status("Server is not running");
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
        self.server_panel_state.set_status("Server stopped");
        self.log("Server stopped");

        // Clear chat model info if chat was using server model
        if self.backend.is_none() {
            self.chat_state.current_model = None;
            self.chat_state.current_backend = None;
        }
    }

    async fn load_additional_model(&mut self) {
        if self.server_panel_state.phase != ServerPhase::Running {
            self.server_panel_state.set_error("Start the server first");
            return;
        }

        // Don't start if already loading
        if self.additional_model_load_task.is_some() {
            self.server_panel_state
                .set_status("Already loading a model, please wait...");
            return;
        }

        let model_path = match self.server_panel_state.selected_model_path() {
            Some(p) => p,
            None => {
                self.server_panel_state
                    .set_error("No model selected. Use Left/Right to pick a model.");
                return;
            }
        };

        // Reject multimodal projector files
        let model_name_lower = std::path::Path::new(&model_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        if model_name_lower.contains("mmproj") {
            self.server_panel_state.set_error(
                "Error: This is a multimodal projector (-mmproj), not a standalone model. \
                 Select the main model file (e.g. Q4_K_M.gguf) instead.",
            );
            return;
        }

        let model_name = std::path::Path::new(&model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();

        self.server_panel_state.phase = ServerPhase::LoadingModel;
        self.server_panel_state
            .set_status(format!("Loading additional model: {}...", model_name));
        self.additional_model_name_hint = Some(model_name.clone());

        // If detached server, use HTTP API to load model
        if let Some(ref state) = self.server_state {
            let host = self.connect_host();
            let port = state.port;
            let api_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());
            let gpu_layers = self.server_panel_state.gpu_layers;
            let context_size = self.server_panel_state.context_size;
            let batch_size = self.server_panel_state.batch_size;

            let task = tokio::spawn(async move {
                let client = http_client(120)?;
                let url = format!("http://{}:{}/v1/models/load", host, port);

                let body = serde_json::json!({
                    "model_path": model_path,
                    "gpu_layers": gpu_layers,
                    "context_size": context_size,
                    "batch_size": batch_size,
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
            let mut backend =
                BackendFactory::create_for_model(backend_type, &hardware, &load_config.model_path)
                    .map_err(|e| {
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

                    self.server_panel_state.set_status(format!(
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
                self.server_panel_state
                    .set_error(format!("Failed to load '{}': {}", name, e));
            }
            Ok(AdditionalModelLoadResult::Detached(Ok(model_id))) => {
                let model_name = hint.unwrap_or_else(|| "unknown".to_string());
                self.server_panel_state.set_status(format!(
                    "Loaded '{}' on remote server (id: {})",
                    model_name, model_id
                ));
                // Fetch updated model list from detached server
                self.refresh_remote_loaded_models().await;
                self.server_panel_state.phase = ServerPhase::Running;
            }
            Ok(AdditionalModelLoadResult::Detached(Err(e))) => {
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state
                    .set_error(format!("Failed to load model on remote server: {}", e));
            }
            Err(e) => {
                self.server_panel_state.phase = ServerPhase::Running;
                self.server_panel_state
                    .set_error(format!("Model loading task crashed: {}", e));
            }
        }
    }

    async fn unload_model_action(&mut self) {
        if self.server_panel_state.loaded_models.is_empty() {
            self.server_panel_state.set_error("No models to unload");
            return;
        }

        let model_id = match self.server_panel_state.selected_unload_model_id() {
            Some(id) => id,
            None => return,
        };

        // Detached server: use HTTP API
        if let Some(ref state) = self.server_state {
            let host = self.connect_host();
            let port = state.port;
            let api_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());
            let mid = model_id.clone();

            let result = tokio::spawn(async move {
                let client = http_client(15)?;
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
                    self.server_panel_state
                        .set_status(format!("Unloaded model: {}", model_id));
                }
                Ok(Err(e)) => {
                    self.server_panel_state.set_error(format!("Error: {}", e));
                }
                Err(e) => {
                    self.server_panel_state
                        .set_error(format!("Task crashed: {}", e));
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

                    self.server_panel_state
                        .set_status(format!("Unloaded model: {}", model_id));
                }
                Err(e) => {
                    self.server_panel_state.set_error(format!("Error: {}", e));
                }
            }
        }
    }

    async fn set_default_model_action(&mut self) {
        if self.server_panel_state.loaded_models.is_empty() {
            self.server_panel_state.set_error("No models loaded");
            return;
        }

        let model_id = match self.server_panel_state.selected_default_model_id() {
            Some(id) => id,
            None => return,
        };

        // Detached server: use HTTP API
        if let Some(ref state) = self.server_state {
            let host = self.connect_host();
            let port = state.port;
            let api_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());
            let mid = model_id.clone();

            let result = tokio::spawn(async move {
                let client = http_client(15)?;
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
                    self.server_panel_state
                        .set_status(format!("Default model set to: {}", model_id));
                }
                Ok(Err(e)) => {
                    self.server_panel_state.set_error(format!("Error: {}", e));
                }
                Err(e) => {
                    self.server_panel_state
                        .set_error(format!("Task crashed: {}", e));
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

                    self.server_panel_state
                        .set_status(format!("Default model set to: {}", model_id));
                }
                Err(e) => {
                    self.server_panel_state.set_error(format!("Error: {}", e));
                }
            }
        }
    }

    /// Fetch the list of loaded models from a detached server via GET /v1/models
    async fn refresh_remote_loaded_models(&mut self) {
        let Some(ref state) = self.server_state else {
            return;
        };
        let host = self.connect_host();
        let port = state.port;
        let api_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());

        let result = tokio::spawn(async move {
            let client = http_client(15)?;
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

    /// Keep the server panel's memory numbers fresh: while the server tab
    /// is open, re-read OS memory every 5s on a blocking thread (file I/O
    /// must not run on the async runtime — it would freeze the TUI).
    async fn poll_mem_refresh(&mut self) {
        // Collect a finished refresh first
        if let Some(task) = self.mem_refresh_task.take() {
            if task.is_finished() {
                if let Ok((total, avail)) = task.await {
                    if total > 0 {
                        self.hardware.memory_total_mb = total;
                        self.server_panel_state.hardware.memory_total_mb = total;
                    }
                    if avail > 0 {
                        self.hardware.memory_available_mb = avail;
                        self.server_panel_state.hardware.memory_available_mb = avail;
                    }
                }
            } else {
                self.mem_refresh_task = Some(task);
                return; // still running — don't spawn another
            }
        }

        if self.mode == AppMode::Server
            && self.last_mem_refresh.elapsed() >= std::time::Duration::from_secs(5)
        {
            self.last_mem_refresh = std::time::Instant::now();
            self.mem_refresh_task =
                Some(tokio::task::spawn_blocking(athenas_core::detect_memory_mb));
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
                        self.server_panel_state.set_status("Server stopped");
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
                        self.server_panel_state
                            .set_error(format!("Server error: {}", e));
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
                KeyCode::Tab | KeyCode::Down => {
                    self.api_key_modal.next_form_field();
                }
                KeyCode::BackTab | KeyCode::Up => {
                    self.api_key_modal.prev_form_field();
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
        let host = self.connect_host();
        let port = state.port;
        let admin_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());
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
        let host = self.connect_host();
        let port = state.port;
        let admin_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());
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
        let host = self.connect_host();
        let port = state.port;
        let admin_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());
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
            self.server_panel_state
                .set_error("Server not running — start it first");
            return;
        };
        let host = self.connect_host();
        let port = state.port;
        let api_key = self.server_panel_state.auth_bearer().map(|s| s.to_string());

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
                            usage_requests: k
                                .get("usage")
                                .and_then(|u| u.get("requests"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            usage_tokens_prompt: k
                                .get("usage")
                                .and_then(|u| u.get("tokens_prompt"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            usage_tokens_generated: k
                                .get("usage")
                                .and_then(|u| u.get("tokens_generated"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            usage_tokens_total: k
                                .get("usage")
                                .and_then(|u| u.get("tokens_total"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            usage_date: k
                                .get("usage")
                                .and_then(|u| u.get("date"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            rate_limit_remaining: k
                                .get("usage")
                                .and_then(|u| u.get("rate_limit_remaining"))
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0)
                                as u32,
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
                self.server_panel_state
                    .set_status(format!("Loaded {} API key(s)", count));
                // Also update the modal if it's open
                self.api_key_modal.set_keys(keys);
                // Provision the TUI's own key if the server enforces auth
                self.ensure_tui_key().await;
            }
            Ok(Err(e)) => {
                let msg = format!("Failed to fetch API keys: {}", e);
                self.server_panel_state.set_error(msg.clone());
                if self.api_key_modal.open {
                    self.api_key_modal.set_error(msg);
                }
            }
            Err(e) => {
                let msg = format!("Request failed: {}", e);
                self.server_panel_state.set_error(msg.clone());
                if self.api_key_modal.open {
                    self.api_key_modal.set_error(msg);
                }
            }
        }
    }

    /// Ensure the TUI has a working API key for TUI→server auth.
    ///
    /// The server only returns masked keys in listings, so the TUI keeps
    /// its own key persisted in ~/.athenas/tui_key.json. This runs after
    /// refresh_api_keys():
    /// - Zero keys on the server = bootstrap mode, no key needed.
    /// - Stored key_id still present and active = reuse it.
    /// - Otherwise create a new "athenas-tui" key (loopback is allowed
    ///   to manage keys) and persist it.
    async fn ensure_tui_key(&mut self) {
        if self.server_panel_state.api_keys.is_empty() {
            // Bootstrap mode — server accepts unauthenticated requests.
            self.server_panel_state.tui_api_key = None;
            self.server_panel_state.tui_key_id = None;
            return;
        }

        // Load the persisted key if we don't have one in memory.
        if self.server_panel_state.tui_api_key.is_none() {
            if let Some((key_id, api_key)) = crate::server_panel::load_tui_key() {
                self.server_panel_state.tui_key_id = Some(key_id);
                self.server_panel_state.tui_api_key = Some(api_key);
            }
        }

        // Reuse the stored key if it's still listed and active.
        if let Some(ref kid) = self.server_panel_state.tui_key_id {
            let still_valid = self
                .server_panel_state
                .api_keys
                .iter()
                .any(|k| &k.key_id == kid && k.active);
            if still_valid {
                return;
            }
        }

        // Create a fresh key via loopback admin access.
        let Some(ref state) = self.server_state else {
            return;
        };
        let host = self.connect_host();
        let port = state.port;

        let result = tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .map_err(|e| e.to_string())?;
            let url = format!("http://{}:{}/v1/keys", host, port);
            let body = serde_json::json!({
                "name": "athenas-tui",
                "rate_limit_per_minute": 0,
                "daily_token_limit": 0,
                "allowed_models": [],
            });
            let resp = client
                .post(&url)
                .json(&body)
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
                let key_id = json
                    .get("key_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let api_key = json
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if !key_id.is_empty() && !api_key.is_empty() {
                    crate::server_panel::save_tui_key(&key_id, &api_key);
                    self.server_panel_state.tui_key_id = Some(key_id);
                    self.server_panel_state.tui_api_key = Some(api_key);
                    tracing::info!("Provisioned TUI API key");
                }
            }
            Ok(Err(e)) => {
                tracing::warn!("Failed to provision TUI API key: {}", e);
            }
            Err(e) => {
                tracing::warn!("TUI key provisioning request failed: {}", e);
            }
        }
    }
}
