use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::server_panel::ApiKeyInfo;

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
    /// Whether the status is an error (red) or info (cyan)
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
                // Submit — caller will send the request
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

/// Mask an API key, showing only the last 8 characters.
/// e.g. "sk-ath-abcdef1234567890" -> "sk-ath-••••••••••••••••56789012"
pub fn mask_key(key: &str) -> String {
    if key.len() <= 12 {
        return "••••".to_string();
    }
    let prefix_end = key.find('-').map(|i| i + 1).unwrap_or(0);
    let prefix = &key[..prefix_end.min(key.len())];
    let suffix = &key[key.len().saturating_sub(8)..];
    format!("{}••••••••{}", prefix, suffix)
}

/// Render the API Key modal as a centered overlay.
pub fn render_api_key_modal(f: &mut Frame, state: &ApiKeyModalState) {
    if !state.open {
        return;
    }

    let area = f.area();

    // Calculate modal size — centered, 70% width, 80% height
    let modal_width = (area.width as f32 * 0.7) as u16;
    let modal_height = (area.height as f32 * 0.8) as u16;
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
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " API Key Management ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::Cyan));

    // Render the outer border first, then split the inner area
    f.render_widget(block, area);
    let inner = area.inner(ratatui::layout::Margin::new(1, 1));

    // Layout: title hint + key list + details + actions + status
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Header hint
            Constraint::Min(5),    // Key list
            Constraint::Length(8), // Selected key details
            Constraint::Length(4), // Action hints
            Constraint::Length(2), // Status
        ])
        .split(inner);

    // Header hint
    let header = Paragraph::new(Line::from(vec![Span::styled(
        "Manage API keys for multi-tenant access",
        Style::default().fg(Color::DarkGray),
    )]))
    .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);

    // Key list
    let list_block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" Keys ", Style::default().fg(Color::Yellow)))
        .border_style(Style::default().fg(Color::DarkGray));

    if state.keys.is_empty() {
        let empty = Paragraph::new(Line::from(vec![
            Span::styled(
                "\n  No API keys loaded",
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("\n\n  Press ", Style::default().fg(Color::Gray)),
            Span::styled(
                "n",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to create a new key", Style::default().fg(Color::Gray)),
        ]))
        .block(list_block);
        f.render_widget(empty, chunks[1]);
    } else {
        let mut lines: Vec<Line> = Vec::new();
        for (i, k) in state.keys.iter().enumerate() {
            let is_selected = i == state.selected;
            let status_icon = if k.active { "●" } else { "○" };
            let status_color = if k.active { Color::Green } else { Color::Red };
            let status_text = if k.active { "active" } else { "revoked" };
            let masked = mask_key(&k.api_key);

            let prefix = if is_selected { "▶" } else { " " };

            let style = if is_selected {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };

            lines.push(Line::from(vec![
                Span::styled(format!(" {} ", prefix), style),
                Span::styled(format!("{:<20}", k.name), style),
                Span::styled(
                    format!("  {:<24}", masked),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(status_icon, Style::default().fg(status_color)),
                Span::styled(
                    format!(" {}", status_text),
                    Style::default().fg(status_color),
                ),
            ]));
        }
        let list = Paragraph::new(lines).block(list_block);
        f.render_widget(list, chunks[1]);
    }

    // Selected key details
    let detail_block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Details ",
            Style::default().fg(Color::Yellow),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    if let Some(k) = state.selected_key() {
        let active_color = if k.active { Color::Green } else { Color::Red };
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

        let details = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("  Name:           ", Style::default().fg(Color::DarkGray)),
                Span::styled(&k.name, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Key ID:         ", Style::default().fg(Color::DarkGray)),
                Span::styled(&k.key_id, Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled("  Key:            ", Style::default().fg(Color::DarkGray)),
                Span::styled(mask_key(&k.api_key), Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("  Status:         ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    if k.active { "active" } else { "revoked" },
                    Style::default().fg(active_color),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Rate limit:     ", Style::default().fg(Color::DarkGray)),
                Span::styled(rate_str, Style::default().fg(Color::White)),
                Span::styled("   Token limit: ", Style::default().fg(Color::DarkGray)),
                Span::styled(token_str, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Allowed models: ", Style::default().fg(Color::DarkGray)),
                Span::styled(models_str, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Created:        ", Style::default().fg(Color::DarkGray)),
                Span::styled(&k.created_at, Style::default().fg(Color::Gray)),
            ]),
        ])
        .block(detail_block);
        f.render_widget(details, chunks[2]);
    } else {
        let empty = Paragraph::new("").block(detail_block);
        f.render_widget(empty, chunks[2]);
    }

    // Action hints
    let actions = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                "  ↑/↓",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Select   ", Style::default().fg(Color::Gray)),
            Span::styled(
                "n",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" New   ", Style::default().fg(Color::Gray)),
            Span::styled(
                "r",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Revoke   ", Style::default().fg(Color::Gray)),
            Span::styled(
                "d",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Delete   ", Style::default().fg(Color::Gray)),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Refresh   ", Style::default().fg(Color::Gray)),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Close", Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
    ])
    .block(Block::default().borders(Borders::NONE));
    f.render_widget(actions, chunks[3]);

    // Status
    if let Some(ref msg) = state.status_message {
        let color = if state.status_is_error {
            Color::Red
        } else {
            Color::Cyan
        };
        let status = Paragraph::new(Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(msg, Style::default().fg(color)),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(status, chunks[4]);
    }
}

fn render_create_form_phase(f: &mut Frame, area: Rect, state: &ApiKeyModalState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Create New API Key ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::Green));

    f.render_widget(block, area);
    let inner = area.inner(ratatui::layout::Margin::new(1, 1));

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Header
            Constraint::Min(10),   // Form fields
            Constraint::Length(3), // Hints
            Constraint::Length(2), // Status
        ])
        .split(inner);

    // Header
    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            "Fill in the fields below — press ",
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            "Enter",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" to advance", Style::default().fg(Color::DarkGray)),
    ]))
    .alignment(Alignment::Center);
    f.render_widget(header, chunks[0]);

    // Form fields
    let form_block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" Form ", Style::default().fg(Color::Yellow)))
        .border_style(Style::default().fg(Color::DarkGray));

    let fields: [(&str, &String, CreateFormField, &str); 4] = [
        (
            "Name",
            &state.form_name,
            CreateFormField::Name,
            "Descriptive name for this key",
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
            "Comma-separated, empty = all",
        ),
    ];

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(""));

    for (label, value, field, hint) in &fields {
        let is_current = state.form_field.as_ref() == Some(field);
        let prefix = if is_current { "► " } else { "  " };
        let display = if is_current {
            format!("{}│", value)
        } else if value.is_empty() {
            "(empty)".to_string()
        } else {
            value.to_string()
        };
        let style = if is_current {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let label_style = if is_current {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("  {}{: <20}: ", prefix, label), label_style),
            Span::styled(display, style),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      ", Style::default()),
            Span::styled(*hint, Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(""));
    }

    let form = Paragraph::new(lines).block(form_block);
    f.render_widget(form, chunks[1]);

    // Hints
    let hints = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            "  Enter",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" = Next/Submit   ", Style::default().fg(Color::Gray)),
        Span::styled(
            "Esc",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" = Cancel", Style::default().fg(Color::Gray)),
    ])])
    .alignment(Alignment::Center);
    f.render_widget(hints, chunks[2]);

    // Status
    if let Some(ref msg) = state.status_message {
        let color = if state.status_is_error {
            Color::Red
        } else {
            Color::Cyan
        };
        let status = Paragraph::new(Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(msg, Style::default().fg(color)),
        ]))
        .alignment(Alignment::Center);
        f.render_widget(status, chunks[3]);
    }
}

fn render_key_revealed_phase(f: &mut Frame, area: Rect, state: &ApiKeyModalState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Key Created — Copy Now! ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::Yellow));

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Warning
            Constraint::Min(5),    // Key display
            Constraint::Length(3), // Details
            Constraint::Length(3), // Hint
        ])
        .split(area);

    // Warning
    let warning = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "⚠  ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "This is the only time the full key will be shown.",
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![Span::styled(
            "    Copy it now — it cannot be retrieved later.",
            Style::default().fg(Color::Red),
        )]),
    ])
    .alignment(Alignment::Center);
    f.render_widget(warning, chunks[0]);

    // Key display
    if let Some((ref name, ref key)) = state.revealed_key {
        let key_block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                " Your New API Key ",
                Style::default().fg(Color::Green),
            ))
            .border_style(Style::default().fg(Color::Green));

        let key_display = Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Name: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    name,
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Key:  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    key,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ])
        .block(key_block);
        f.render_widget(key_display, chunks[1]);

        // Details
        let details = Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Rate limit: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}/min", state.form_rate_limit),
                    Style::default().fg(Color::White),
                ),
                Span::styled("   Token limit: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{}/day", state.form_token_limit),
                    Style::default().fg(Color::White),
                ),
            ]),
        ])
        .alignment(Alignment::Center);
        f.render_widget(details, chunks[2]);
    }

    // Hint
    let hint = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Press ", Style::default().fg(Color::Gray)),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" or ", Style::default().fg(Color::Gray)),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to continue", Style::default().fg(Color::Gray)),
        ]),
    ])
    .alignment(Alignment::Center);
    f.render_widget(hint, chunks[3]);

    f.render_widget(block, area);
}

fn render_confirm_phase(f: &mut Frame, area: Rect, state: &ApiKeyModalState, action: &str) {
    let (title, title_color, action_desc) = match action {
        "revoke" => (" Confirm Revoke ", Color::Yellow, "revoke (deactivate)"),
        "delete" => (" Confirm Delete ", Color::Red, "permanently delete"),
        _ => return,
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            title,
            Style::default()
                .fg(title_color)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(title_color));

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Warning
            Constraint::Min(4),    // Key info
            Constraint::Length(3), // Confirm/Cancel
        ])
        .split(area);

    // Warning
    let warning_text = if action == "delete" {
        "This action cannot be undone. The key will be permanently removed."
    } else {
        "The key will be deactivated. You can re-enable it later by creating a new one."
    };
    let warning = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "⚠  ",
                Style::default()
                    .fg(title_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(warning_text, Style::default().fg(title_color)),
        ]),
    ])
    .alignment(Alignment::Center);
    f.render_widget(warning, chunks[0]);

    // Key info
    if let Some(k) = state.selected_key() {
        let key_block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                " Selected Key ",
                Style::default().fg(Color::DarkGray),
            ))
            .border_style(Style::default().fg(Color::DarkGray));

        let info = Paragraph::new(vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Name:  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    &k.name,
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Key:   ", Style::default().fg(Color::DarkGray)),
                Span::styled(mask_key(&k.api_key), Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("  ID:    ", Style::default().fg(Color::DarkGray)),
                Span::styled(&k.key_id, Style::default().fg(Color::Gray)),
            ]),
        ])
        .block(key_block);
        f.render_widget(info, chunks[1]);
    }

    // Confirm/Cancel
    let confirm = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  Enter",
                Style::default()
                    .fg(title_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" = {}   ", action_desc),
                Style::default().fg(Color::Gray),
            ),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" = Cancel", Style::default().fg(Color::Gray)),
        ]),
    ])
    .alignment(Alignment::Center);
    f.render_widget(confirm, chunks[2]);

    f.render_widget(block, area);
}
