use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
    Frame,
};

use crate::server_panel::ApiKeyInfo;
use crate::theme;

/// Which phase of the API key modal is active.
#[derive(Debug, Clone, PartialEq)]
pub enum ApiKeyModalPhase {
    /// Main list view — shows all keys, allows selection and actions
    List,
    /// Creating a new key — form is active
    CreateForm,
    /// A key was just created — show the full key ONCE with a copy warning
    KeyRevealed,
    /// Confirm revoke dialog
    ConfirmRevoke,
    /// Confirm delete dialog
    ConfirmDelete,
}

/// Which field of the create form is being edited.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateFormField {
    Name,
    RateLimit,
    TokenLimit,
    AllowedModels,
}

/// State for the API Key management modal.
pub struct ApiKeyModalState {
    pub open: bool,
    pub phase: ApiKeyModalPhase,
    pub keys: Vec<ApiKeyInfo>,
    pub selected: usize,

    // Create form fields
    pub form_field: Option<CreateFormField>,
    pub form_name: String,
    pub form_rate_limit: String,
    pub form_token_limit: String,
    pub form_allowed_models: String,

    // Revealed key (shown once after creation)
    pub revealed_key: Option<(String, String)>, // (key_name, full_key)

    // Status / error message
    pub status_message: Option<String>,
    /// Whether the status is an error (rose) or info (primary)
    pub status_is_error: bool,
}

impl Default for ApiKeyModalState {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiKeyModalState {
    pub fn new() -> Self {
        Self {
            open: false,
            phase: ApiKeyModalPhase::List,
            keys: Vec::new(),
            selected: 0,
            form_field: None,
            form_name: String::new(),
            form_rate_limit: "60".to_string(),
            form_token_limit: "0".to_string(),
            form_allowed_models: String::new(),
            revealed_key: None,
            status_message: None,
            status_is_error: false,
        }
    }

    pub fn open(&mut self) {
        self.open = true;
        self.phase = ApiKeyModalPhase::List;
        self.form_field = None;
        self.revealed_key = None;
        self.status_message = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.phase = ApiKeyModalPhase::List;
        self.form_field = None;
        self.revealed_key = None;
        self.status_message = None;
    }

    pub fn set_keys(&mut self, keys: Vec<ApiKeyInfo>) {
        let count = keys.len();
        self.keys = keys;
        if self.selected >= self.keys.len() && !self.keys.is_empty() {
            self.selected = self.keys.len() - 1;
        }
        if self.keys.is_empty() {
            self.selected = 0;
        }
        if self.phase == ApiKeyModalPhase::List {
            self.status_message = Some(format!("Loaded {} key(s)", count));
            self.status_is_error = false;
        }
    }

    pub fn set_error(&mut self, msg: String) {
        self.status_message = Some(msg);
        self.status_is_error = true;
    }

    pub fn set_info(&mut self, msg: String) {
        self.status_message = Some(msg);
        self.status_is_error = false;
    }

    pub fn selected_key(&self) -> Option<&ApiKeyInfo> {
        self.keys.get(self.selected)
    }

    pub fn select_next(&mut self) {
        if !self.keys.is_empty() {
            self.selected = (self.selected + 1) % self.keys.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.keys.is_empty() {
            if self.selected == 0 {
                self.selected = self.keys.len() - 1;
            } else {
                self.selected -= 1;
            }
        }
    }

    /// Start the create form — resets all fields.
    pub fn start_create(&mut self) {
        self.phase = ApiKeyModalPhase::CreateForm;
        self.form_field = Some(CreateFormField::Name);
        self.form_name.clear();
        self.form_rate_limit = "60".to_string();
        self.form_token_limit = "0".to_string();
        self.form_allowed_models.clear();
        self.status_message = None;
    }

    /// Cancel the create form — return to list.
    pub fn cancel_create(&mut self) {
        self.phase = ApiKeyModalPhase::List;
        self.form_field = None;
        self.status_message = None;
    }

    /// Move focus to the next field in the create form (Tab / Down).
    pub fn next_form_field(&mut self) {
        self.form_field = match self.form_field {
            Some(CreateFormField::Name) => Some(CreateFormField::RateLimit),
            Some(CreateFormField::RateLimit) => Some(CreateFormField::TokenLimit),
            Some(CreateFormField::TokenLimit) => Some(CreateFormField::AllowedModels),
            Some(CreateFormField::AllowedModels) | None => Some(CreateFormField::Name),
        };
        self.status_message = None;
    }

    /// Move focus to the previous field in the create form (Shift+Tab / Up).
    pub fn prev_form_field(&mut self) {
        self.form_field = match self.form_field {
            Some(CreateFormField::Name) | None => Some(CreateFormField::AllowedModels),
            Some(CreateFormField::RateLimit) => Some(CreateFormField::Name),
            Some(CreateFormField::TokenLimit) => Some(CreateFormField::RateLimit),
            Some(CreateFormField::AllowedModels) => Some(CreateFormField::TokenLimit),
        };
        self.status_message = None;
    }

    /// Advance to the next form field, or submit if on the last field.
    /// Returns true if the form was submitted (caller should send the request).
    pub fn advance_form(&mut self) -> bool {
        let current = self.form_field.clone();
        match current {
            Some(CreateFormField::Name) => {
                if self.form_name.trim().is_empty() {
                    self.set_error("Name cannot be empty".to_string());
                    return false;
                }
                self.form_field = Some(CreateFormField::RateLimit);
                self.status_message = None;
                false
            }
            Some(CreateFormField::RateLimit) => {
                self.form_field = Some(CreateFormField::TokenLimit);
                self.status_message = None;
                false
            }
            Some(CreateFormField::TokenLimit) => {
                self.form_field = Some(CreateFormField::AllowedModels);
                self.status_message = None;
                false
            }
            Some(CreateFormField::AllowedModels) => {
                if self.form_name.trim().is_empty() {
                    self.form_field = Some(CreateFormField::Name);
                    self.set_error("Name cannot be empty".to_string());
                    return false;
                }
                self.form_field = None;
                true
            }
            None => false,
        }
    }

    /// Handle a character input in the create form.
    pub fn form_input_char(&mut self, c: char) {
        let Some(ref field) = self.form_field else {
            return;
        };
        match field {
            CreateFormField::Name => self.form_name.push(c),
            CreateFormField::RateLimit => {
                if c.is_ascii_digit() {
                    self.form_rate_limit.push(c);
                }
            }
            CreateFormField::TokenLimit => {
                if c.is_ascii_digit() {
                    self.form_token_limit.push(c);
                }
            }
            CreateFormField::AllowedModels => self.form_allowed_models.push(c),
        }
    }

    /// Handle backspace in the create form.
    pub fn form_backspace(&mut self) {
        let Some(ref field) = self.form_field else {
            return;
        };
        match field {
            CreateFormField::Name => {
                self.form_name.pop();
            }
            CreateFormField::RateLimit => {
                self.form_rate_limit.pop();
            }
            CreateFormField::TokenLimit => {
                self.form_token_limit.pop();
            }
            CreateFormField::AllowedModels => {
                self.form_allowed_models.pop();
            }
        }
    }

    /// Show the revealed key after successful creation.
    pub fn reveal_key(&mut self, key_name: String, full_key: String) {
        self.revealed_key = Some((key_name, full_key));
        self.phase = ApiKeyModalPhase::KeyRevealed;
        self.status_message = None;
    }

    /// Dismiss the revealed key and return to list.
    pub fn dismiss_revealed(&mut self) {
        self.revealed_key = None;
        self.phase = ApiKeyModalPhase::List;
    }

    /// Start the revoke confirmation for the selected key.
    pub fn start_revoke(&mut self) -> bool {
        if let Some(k) = self.selected_key() {
            if k.active {
                self.phase = ApiKeyModalPhase::ConfirmRevoke;
                self.status_message = None;
                true
            } else {
                self.set_error("Key is already revoked".to_string());
                false
            }
        } else {
            self.set_error("No key selected".to_string());
            false
        }
    }

    /// Start the delete confirmation for the selected key.
    pub fn start_delete(&mut self) -> bool {
        if self.selected_key().is_some() {
            self.phase = ApiKeyModalPhase::ConfirmDelete;
            self.status_message = None;
            true
        } else {
            self.set_error("No key selected".to_string());
            false
        }
    }

    /// Cancel a confirmation dialog — return to list.
    pub fn cancel_confirm(&mut self) {
        self.phase = ApiKeyModalPhase::List;
        self.status_message = None;
    }

    /// Get the key_id of the selected key (for revoke/delete actions).
    pub fn selected_key_id(&self) -> Option<String> {
        self.selected_key().map(|k| k.key_id.clone())
    }

    /// Get the form values for submission.
    pub fn form_values(&self) -> (String, u32, u64, Vec<String>) {
        let name = self.form_name.trim().to_string();
        let rate_limit: u32 = self.form_rate_limit.trim().parse().unwrap_or(60);
        let token_limit: u64 = self.form_token_limit.trim().parse().unwrap_or(0);
        let allowed_models: Vec<String> = if self.form_allowed_models.trim().is_empty() {
            Vec::new()
        } else {
            self.form_allowed_models
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };
        (name, rate_limit, token_limit, allowed_models)
    }
}

/// Mask an API key safely (UTF-8 char-aware), showing prefix up to first '-' and last 8 chars.
/// e.g. "sk-ath-abcdef1234567890" -> "sk-••••••••34567890"
pub fn mask_key(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 12 {
        return "••••".to_string();
    }
    let prefix_end = chars
        .iter()
        .position(|&c| c == '-')
        .map(|i| i + 1)
        .unwrap_or(0)
        .min(chars.len());
    let prefix: String = chars[..prefix_end].iter().collect();
    let suffix_start = chars.len().saturating_sub(8);
    let suffix: String = chars[suffix_start..].iter().collect();
    format!("{}••••••••{}", prefix, suffix)
}

/// Render the API Key modal as a centered overlay.
pub fn render_api_key_modal(f: &mut Frame, state: &ApiKeyModalState) {
    if !state.open {
        return;
    }

    let area = f.area();

    // Adaptive size — comfortably fits 80x24 and scales gracefully on large screens
    let modal_width = ((area.width as f32 * 0.86) as u16)
        .clamp(60.min(area.width), area.width.saturating_sub(2).max(1));
    let modal_height = ((area.height as f32 * 0.86) as u16)
        .clamp(18.min(area.height), area.height.saturating_sub(2).max(1));
    let modal_x = (area.width.saturating_sub(modal_width)) / 2;
    let modal_y = (area.height.saturating_sub(modal_height)) / 2;
    let modal_area = Rect::new(modal_x, modal_y, modal_width, modal_height);

    // Clear the background behind the modal
    f.render_widget(Clear, modal_area);

    match state.phase {
        ApiKeyModalPhase::List => render_list_phase(f, modal_area, state),
        ApiKeyModalPhase::CreateForm => render_create_form_phase(f, modal_area, state),
        ApiKeyModalPhase::KeyRevealed => render_key_revealed_phase(f, modal_area, state),
        ApiKeyModalPhase::ConfirmRevoke => render_confirm_phase(f, modal_area, state, "revoke"),
        ApiKeyModalPhase::ConfirmDelete => render_confirm_phase(f, modal_area, state, "delete"),
    }
}

fn render_list_phase(f: &mut Frame, area: Rect, state: &ApiKeyModalState) {
    let block = theme::rounded_block_colored(" ◆ API Key Management ", theme::PRIMARY)
        .style(Style::default().bg(theme::BG_SURFACE));

    // Render the outer border first, then split block.inner(area)
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),    // Main content (2-column or stacked)
            Constraint::Length(1), // Action hints
            Constraint::Length(1), // Status
        ])
        .split(inner);

    let (list_area, detail_area) = if inner.width >= 68 {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
            .split(chunks[0]);
        (cols[0], cols[1])
    } else {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(chunks[0]);
        (rows[0], rows[1])
    };

    // Left: Key list (scrollable)
    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(Span::styled(
            format!(" Keys ({}) ", state.keys.len()),
            Style::default()
                .fg(theme::WARNING)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(theme::BORDER));

    if state.keys.is_empty() {
        let empty = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No API keys configured",
                Style::default().fg(theme::TEXT_MUTED),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Press ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    " n ",
                    Style::default()
                        .fg(theme::SUCCESS)
                        .bg(theme::BG_HIGHLIGHT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" to create a new key", Style::default().fg(theme::TEXT_DIM)),
            ]),
        ])
        .block(list_block);
        f.render_widget(empty, list_area);
    } else {
        let mut lines: Vec<Line> = Vec::with_capacity(state.keys.len());
        for (i, k) in state.keys.iter().enumerate() {
            let is_selected = i == state.selected;
            let status_icon = if k.active { "●" } else { "○" };
            let status_color = if k.active {
                theme::SUCCESS
            } else {
                theme::ERROR
            };
            let prefix = if is_selected { "▶ " } else { "  " };

            let row_style = if is_selected {
                Style::default()
                    .fg(theme::TEXT)
                    .bg(theme::BG_HIGHLIGHT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::TEXT_DIM)
            };

            let name_display: String = k.name.chars().take(18).collect();
            lines.push(Line::from(vec![
                Span::styled(prefix, row_style),
                Span::styled(
                    status_icon,
                    Style::default().fg(status_color).bg(if is_selected {
                        theme::BG_HIGHLIGHT
                    } else {
                        theme::BG_SURFACE
                    }),
                ),
                Span::styled(format!(" {:<18}", name_display), row_style),
            ]));
        }

        let visible_h = list_area.height.saturating_sub(2) as usize;
        let scroll = if visible_h > 0 && state.selected >= visible_h {
            (state.selected - visible_h + 1) as u16
        } else {
            0
        };
        let list = Paragraph::new(lines).block(list_block).scroll((scroll, 0));
        f.render_widget(list, list_area);
        theme::render_scrollbar(f, list_area, state.keys.len(), state.selected);
    }

    // Right: Selected key details
    let detail_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(Span::styled(
            " Details & Usage ",
            Style::default()
                .fg(theme::PRIMARY)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(theme::BORDER));

    if let Some(k) = state.selected_key() {
        let active_color = if k.active {
            theme::SUCCESS
        } else {
            theme::ERROR
        };
        let models_str = if k.allowed_models.is_empty() {
            "all models".to_string()
        } else {
            k.allowed_models.join(", ")
        };
        let rate_str = if k.rate_limit_per_minute == 0 {
            "unlimited".to_string()
        } else {
            format!("{}/min", k.rate_limit_per_minute)
        };
        let token_str = if k.daily_token_limit == 0 {
            "unlimited".to_string()
        } else {
            format!("{}/day", k.daily_token_limit)
        };
        let remaining_str = if k.rate_limit_per_minute == 0 {
            "unlimited".to_string()
        } else {
            format!("{}/min", k.rate_limit_remaining)
        };

        let details = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(" Name:     ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    &k.name,
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(
                    if k.active {
                        "● active"
                    } else {
                        "○ revoked"
                    },
                    Style::default()
                        .fg(active_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Key:      ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(mask_key(&k.api_key), Style::default().fg(theme::PRIMARY)),
            ]),
            Line::from(vec![
                Span::styled(" ID:       ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(&k.key_id, Style::default().fg(theme::TEXT_DIM)),
            ]),
            Line::from(vec![
                Span::styled(" Limits:   ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(rate_str, Style::default().fg(theme::TEXT)),
                Span::styled(" · ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(token_str, Style::default().fg(theme::TEXT)),
            ]),
            Line::from(vec![
                Span::styled(" Models:   ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(models_str, Style::default().fg(theme::TEXT)),
            ]),
            Line::from(vec![
                Span::styled(" Created:  ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(&k.created_at, Style::default().fg(theme::TEXT_DIM)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                " ── Usage (today) ──",
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled(" Requests: ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    format!("{}", k.usage_requests),
                    Style::default().fg(theme::TEXT),
                ),
                Span::styled("   Remain: ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(remaining_str, Style::default().fg(theme::SUCCESS)),
            ]),
            Line::from(vec![
                Span::styled(" Tokens:   ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    format!("{} total", k.usage_tokens_total),
                    Style::default().fg(theme::PRIMARY),
                ),
                Span::styled(
                    format!(
                        " (in {} / out {})",
                        k.usage_tokens_prompt, k.usage_tokens_generated
                    ),
                    Style::default().fg(theme::TEXT_DIM),
                ),
            ]),
        ])
        .block(detail_block)
        .wrap(Wrap { trim: false });
        f.render_widget(details, detail_area);
    } else {
        let empty = Paragraph::new(Line::styled(
            "  Select a key to view details",
            Style::default().fg(theme::TEXT_MUTED),
        ))
        .block(detail_block);
        f.render_widget(empty, detail_area);
    }

    // Action hints
    let mut hint_spans = Vec::new();
    hint_spans.extend(theme::keycap("↑/↓", "Select"));
    hint_spans.extend(theme::keycap("n", "New"));
    hint_spans.extend(theme::keycap("r", "Revoke"));
    hint_spans.extend(theme::keycap("d", "Delete"));
    hint_spans.extend(theme::keycap("Enter", "Refresh"));
    hint_spans.extend(theme::keycap("Esc", "Close"));
    let actions = Paragraph::new(Line::from(hint_spans)).alignment(Alignment::Center);
    f.render_widget(actions, chunks[1]);

    // Status
    if let Some(ref msg) = state.status_message {
        let color = if state.status_is_error {
            theme::ERROR
        } else {
            theme::SUCCESS
        };
        let status = Paragraph::new(Line::styled(msg.as_str(), Style::default().fg(color)))
            .alignment(Alignment::Center);
        f.render_widget(status, chunks[2]);
    }
}

fn render_create_form_phase(f: &mut Frame, area: Rect, state: &ApiKeyModalState) {
    let block = theme::rounded_block_colored(" ◆ Create New API Key ", theme::SUCCESS)
        .style(Style::default().bg(theme::BG_SURFACE));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Header
            Constraint::Min(10),   // Form fields
            Constraint::Length(1), // Hints
            Constraint::Length(1), // Status
        ])
        .split(inner);

    let header = Paragraph::new(Line::from(vec![
        Span::styled("Fill in fields — ", Style::default().fg(theme::TEXT_DIM)),
        Span::styled(
            "Tab/↑/↓",
            Style::default()
                .fg(theme::PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" navigate, ", Style::default().fg(theme::TEXT_DIM)),
        Span::styled(
            "Enter",
            Style::default()
                .fg(theme::SUCCESS)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" next/submit", Style::default().fg(theme::TEXT_DIM)),
    ]))
    .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);

    let form_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(Span::styled(
            " Key Parameters ",
            Style::default().fg(theme::WARNING),
        ))
        .border_style(Style::default().fg(theme::BORDER));

    let fields: [(&str, &String, CreateFormField, &str); 4] = [
        (
            "Name",
            &state.form_name,
            CreateFormField::Name,
            "Descriptive label for this key (required)",
        ),
        (
            "Rate limit/min",
            &state.form_rate_limit,
            CreateFormField::RateLimit,
            "0 = unlimited",
        ),
        (
            "Daily token limit",
            &state.form_token_limit,
            CreateFormField::TokenLimit,
            "0 = unlimited",
        ),
        (
            "Allowed models",
            &state.form_allowed_models,
            CreateFormField::AllowedModels,
            "Comma-separated IDs, empty = all models",
        ),
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    for (label, value, field, hint) in &fields {
        let is_current = state.form_field.as_ref() == Some(field);
        let prefix = if is_current { "▶ " } else { "  " };
        let display = if is_current {
            format!("{}▏", value)
        } else if value.is_empty() {
            "(empty)".to_string()
        } else {
            value.to_string()
        };
        let value_style = if is_current {
            Style::default()
                .fg(theme::TEXT)
                .bg(theme::BG_HIGHLIGHT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::TEXT_DIM)
        };
        let label_style = if is_current {
            Style::default()
                .fg(theme::SUCCESS)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme::TEXT_MUTED)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("  {}{:<19}: ", prefix, label), label_style),
            Span::styled(display, value_style),
        ]));
        lines.push(Line::from(vec![
            Span::raw("      "),
            Span::styled(*hint, Style::default().fg(theme::TEXT_MUTED)),
        ]));
        lines.push(Line::from(""));
    }

    let form = Paragraph::new(lines).block(form_block);
    f.render_widget(form, chunks[1]);

    let mut hint_spans = Vec::new();
    hint_spans.extend(theme::keycap("Tab/Shift+Tab", "Field"));
    hint_spans.extend(theme::keycap("Enter", "Next / Submit"));
    hint_spans.extend(theme::keycap("Esc", "Cancel"));
    let hints = Paragraph::new(Line::from(hint_spans)).alignment(Alignment::Center);
    f.render_widget(hints, chunks[2]);

    if let Some(ref msg) = state.status_message {
        let color = if state.status_is_error {
            theme::ERROR
        } else {
            theme::PRIMARY
        };
        let status = Paragraph::new(Line::styled(msg.as_str(), Style::default().fg(color)))
            .alignment(Alignment::Center);
        f.render_widget(status, chunks[3]);
    }
}

fn render_key_revealed_phase(f: &mut Frame, area: Rect, state: &ApiKeyModalState) {
    let block = theme::rounded_block_colored(" ⚠ Key Created — Copy Now! ", theme::WARNING)
        .style(Style::default().bg(theme::BG_SURFACE));

    // Render outer border FIRST, then split block.inner(area)
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Warning
            Constraint::Min(5),    // Key display
            Constraint::Length(2), // Details
            Constraint::Length(2), // Hint
        ])
        .split(inner);

    let warning = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                "⚠ ",
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "This is the ONLY time the full key will be shown.",
                Style::default().fg(theme::WARNING),
            ),
        ]),
        Line::from(Span::styled(
            "Copy it now — it cannot be retrieved later.",
            Style::default().fg(theme::ERROR),
        )),
    ])
    .alignment(Alignment::Center);
    f.render_widget(warning, chunks[0]);

    if let Some((ref name, ref key)) = state.revealed_key {
        let key_block = theme::rounded_block_colored(" Your New API Key ", theme::SUCCESS);

        let key_display = Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Name: ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    name,
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Key:  ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    key,
                    Style::default()
                        .fg(theme::PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ])
        .block(key_block)
        .wrap(Wrap { trim: false });
        f.render_widget(key_display, chunks[1]);

        let rate_str = if state.form_rate_limit.trim().parse::<u32>().unwrap_or(0) == 0 {
            "unlimited".to_string()
        } else {
            format!("{}/min", state.form_rate_limit.trim())
        };
        let token_str = if state.form_token_limit.trim().parse::<u64>().unwrap_or(0) == 0 {
            "unlimited".to_string()
        } else {
            format!("{}/day", state.form_token_limit.trim())
        };

        let details = Paragraph::new(Line::from(vec![
            Span::styled("Rate limit: ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(rate_str, Style::default().fg(theme::TEXT)),
            Span::styled("   Token limit: ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(token_str, Style::default().fg(theme::TEXT)),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(details, chunks[2]);
    }

    let mut hint_spans = Vec::new();
    hint_spans.extend(theme::keycap("Enter / Esc", "Continue"));
    let hint = Paragraph::new(Line::from(hint_spans)).alignment(Alignment::Center);
    f.render_widget(hint, chunks[3]);
}

fn render_confirm_phase(f: &mut Frame, area: Rect, state: &ApiKeyModalState, action: &str) {
    let (title, title_color, action_desc) = match action {
        "revoke" => (" ⚠ Confirm Revoke ", theme::WARNING, "revoke (deactivate)"),
        "delete" => (" ⚠ Confirm Delete ", theme::ERROR, "permanently delete"),
        _ => return,
    };

    let block = theme::rounded_block_colored(title, title_color)
        .style(Style::default().bg(theme::BG_SURFACE));

    // Render outer border FIRST, then split block.inner(area)
    let inner = block.inner(area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Warning
            Constraint::Min(5),    // Key info
            Constraint::Length(2), // Confirm/Cancel
        ])
        .split(inner);

    let warning_text = if action == "delete" {
        "This action cannot be undone. The key will be permanently removed."
    } else {
        "The key will be deactivated. You can create a new key later."
    };
    let warning = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "⚠ ",
                Style::default()
                    .fg(title_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(warning_text, Style::default().fg(title_color)),
        ]),
    ])
    .alignment(Alignment::Center)
    .wrap(Wrap { trim: false });
    f.render_widget(warning, chunks[0]);

    if let Some(k) = state.selected_key() {
        let key_block = theme::rounded_block(" Selected Key ", false);
        let info = Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Name:  ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    &k.name,
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Key:   ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(mask_key(&k.api_key), Style::default().fg(theme::PRIMARY)),
            ]),
            Line::from(vec![
                Span::styled("  ID:    ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(&k.key_id, Style::default().fg(theme::TEXT_DIM)),
            ]),
        ])
        .block(key_block);
        f.render_widget(info, chunks[1]);
    }

    let mut spans = Vec::new();
    spans.extend(theme::keycap("Enter", action_desc));
    spans.extend(theme::keycap("Esc", "Cancel"));
    let confirm = Paragraph::new(Line::from(spans)).alignment(Alignment::Center);
    f.render_widget(confirm, chunks[2]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_key_handles_ascii_and_utf8_safely() {
        assert_eq!(mask_key("short"), "••••");
        assert_eq!(mask_key("sk-ath-abcdef1234567890"), "sk-••••••••34567890");
        // Multi-byte UTF-8 characters (e.g. bullets already returned by server or accented chars)
        let masked_from_server = "sk-ath-••••••••1234";
        let res = mask_key(masked_from_server);
        assert!(res.starts_with("sk-••••••••"));
        // Pure multi-byte UTF-8 string longer than 12 chars
        let unicode_key = "chave-αβγδεζηθικλμνξο";
        let res2 = mask_key(unicode_key);
        assert!(res2.starts_with("chave-••••••••"));
        assert!(res2.ends_with("θικλμνξο"));
    }
}
