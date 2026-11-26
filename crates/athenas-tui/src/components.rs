use athenas_core::{HardwareInfo, ModelInfo};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::chat::ChatState;
use crate::log_buffer::{LogLevelFilter, LogsState};
use crate::model_browser::{is_recommended_quant, BrowserPhase, ModelBrowserState};
use crate::model_list::ModelListState;
use crate::server_panel::{ConfigField, ServerPanelState, ServerPhase, ServerSubTab};
use crate::settings::SettingsState;
use crate::theme::{self, fmt_mb, format_bytes_binary};

/// Header bar status info passed from `TuiApp`.
pub struct HeaderStatus<'a> {
    pub active_tab: usize,
    pub server_running: bool,
    pub server_port: Option<u16>,
    pub active_model: Option<&'a str>,
    pub update_version: Option<&'a str>,
}

/// Render the sleek top studio header bar (1 row).
/// Returns the `[start_col, end_col)` of the update pill if rendered (for mouse click hit-testing).
pub fn render_header_bar(
    f: &mut Frame,
    area: Rect,
    status: &HeaderStatus<'_>,
) -> Option<(u16, u16)> {
    let mut left_spans: Vec<Span> = vec![
        Span::styled(
            " ◆ ATHENAS ",
            Style::default()
                .fg(theme::BG_HEADER)
                .bg(theme::PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ];

    let tabs = [
        ("1", "Chat"),
        ("2", "Models"),
        ("3", "Browser"),
        ("4", "Server"),
        ("5", "Settings"),
        ("6", "Logs"),
    ];

    for (i, (num, label)) in tabs.iter().enumerate() {
        let active = i == status.active_tab;
        if active {
            left_spans.push(Span::styled(
                format!(" [{}] {} ", num, label),
                Style::default()
                    .fg(theme::BG_HEADER)
                    .bg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            left_spans.push(Span::styled(
                format!(" [{}]", num),
                Style::default().fg(theme::TEXT_MUTED),
            ));
            left_spans.push(Span::styled(
                format!(" {} ", label),
                Style::default().fg(theme::TEXT_DIM),
            ));
        }
    }

    // Right-aligned live status pills
    let mut right_spans: Vec<Span> = Vec::new();
    let mut update_pill_offset_in_right: Option<(usize, usize)> = None;
    if let Some(ver) = status.update_version {
        let pill_text = format!(" ↑ v{} ", ver);
        let pill_w = pill_text.chars().count();
        let start = right_spans.iter().map(|s| s.content.chars().count()).sum();
        update_pill_offset_in_right = Some((start, start + pill_w));
        right_spans.push(Span::styled(
            pill_text,
            Style::default()
                .fg(theme::BG_HEADER)
                .bg(theme::WARNING)
                .add_modifier(Modifier::BOLD),
        ));
        right_spans.push(Span::raw(" "));
    }

    if let Some(model) = status.active_model {
        let short = short_model_name(model);
        let short_trunc: String = if short.chars().count() > 22 {
            format!("{}…", short.chars().take(21).collect::<String>())
        } else {
            short
        };
        right_spans.push(Span::styled(
            format!(" ◆ {} ", short_trunc),
            Style::default()
                .fg(theme::PRIMARY)
                .bg(theme::BG_HIGHLIGHT)
                .add_modifier(Modifier::BOLD),
        ));
        right_spans.push(Span::raw(" "));
    }

    if status.server_running {
        let srv_label = match status.server_port {
            Some(p) => format!(" ● ONLINE :{} ", p),
            None => " ● ONLINE ".to_string(),
        };
        right_spans.push(Span::styled(
            srv_label,
            Style::default()
                .fg(theme::BG_HEADER)
                .bg(theme::SUCCESS)
                .add_modifier(Modifier::BOLD),
        ));
    } else {
        right_spans.push(Span::styled(
            " ○ OFFLINE ",
            Style::default()
                .fg(theme::TEXT_MUTED)
                .bg(theme::BG_HIGHLIGHT),
        ));
    }
    right_spans.push(Span::raw(" "));
    right_spans.push(Span::styled(
        " [?] Help ",
        Style::default().fg(theme::TEXT_DIM).bg(theme::BG_HIGHLIGHT),
    ));

    let left_len: usize = left_spans.iter().map(|s| s.content.chars().count()).sum();
    let right_len: usize = right_spans.iter().map(|s| s.content.chars().count()).sum();
    let total_w = area.width as usize;

    let mut all_spans = left_spans;
    let mut update_pill_cols = None;
    if total_w > left_len + right_len {
        let pad = total_w - left_len - right_len;
        let right_start = area.x as usize + left_len + pad;
        if let Some((off_start, off_end)) = update_pill_offset_in_right {
            update_pill_cols = Some((
                (right_start + off_start) as u16,
                (right_start + off_end) as u16,
            ));
        }
        all_spans.push(Span::raw(" ".repeat(pad)));
        all_spans.extend(right_spans);
    } else if let Some(ver) = status.update_version {
        let pill_text = format!(" ↑ v{} ", ver);
        let pill_w = pill_text.chars().count();
        if total_w > left_len + pill_w {
            let pad = total_w - left_len - pill_w;
            let start = (area.x as usize + left_len + pad) as u16;
            update_pill_cols = Some((start, start + pill_w as u16));
            all_spans.push(Span::raw(" ".repeat(pad)));
            all_spans.push(Span::styled(
                pill_text,
                Style::default()
                    .fg(theme::BG_HEADER)
                    .bg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ));
        }
    } else if total_w > left_len + 10 {
        // Fit at least the Help badge if narrow
        let help_only = " [?] Help ";
        let pad = total_w.saturating_sub(left_len + help_only.len());
        all_spans.push(Span::raw(" ".repeat(pad)));
        all_spans.push(Span::styled(
            help_only,
            Style::default().fg(theme::TEXT_DIM).bg(theme::BG_HIGHLIGHT),
        ));
    }

    let p = Paragraph::new(Line::from(all_spans)).style(Style::default().bg(theme::BG_HEADER));
    f.render_widget(p, area);
    update_pill_cols
}

/// Map a mouse click column on row 0 to a tab index (0..=5), or `None`.
pub fn header_tab_at_col(col: u16) -> Option<usize> {
    // " ◆ ATHENAS " = 11 cols, " " = 1 col -> tabs start at col 12
    let tabs = [
        ("1", "Chat"),
        ("2", "Models"),
        ("3", "Browser"),
        ("4", "Server"),
        ("5", "Settings"),
        ("6", "Logs"),
    ];
    let mut x = 12u16;
    for (i, (num, label)) in tabs.iter().enumerate() {
        // Each tab is " [N] Label " = 6 + label.len() cols
        let w = (6 + num.len() + label.len()) as u16;
        if col >= x && col < x + w {
            return Some(i);
        }
        x += w;
    }
    None
}

/// Centered Help Modal (`?`).
pub fn render_help_modal(f: &mut Frame, active_tab: usize) {
    let area = f.area();
    let modal_w = 72u16.min(area.width.saturating_sub(4).max(20));
    let modal_h = 24u16.min(area.height.saturating_sub(2).max(12));
    let x = (area.width.saturating_sub(modal_w)) / 2;
    let y = (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect::new(x, y, modal_w, modal_h);

    f.render_widget(Clear, modal_area);

    let block = theme::rounded_block_colored(" ◆ Keyboard Shortcuts & Help ", theme::PRIMARY)
        .style(Style::default().bg(theme::BG_SURFACE));
    let inner = block.inner(modal_area);
    f.render_widget(block, modal_area);

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            " Global Navigation",
            Style::default()
                .fg(theme::WARNING)
                .add_modifier(Modifier::BOLD),
        )),
        help_row("F1–F6 / Alt+1–6", "Switch tabs directly (works everywhere)"),
        help_row("Alt+← / Alt+→", "Cycle tabs prev / next (works everywhere)"),
        help_row("1–6", "Switch tabs (when not typing)"),
        help_row("Tab / Shift+Tab", "Cycle tabs forward / backward"),
        help_row("U", "Open Update Checker (or /update in Chat)"),
        help_row("? / Esc", "Toggle / close help modal"),
        help_row("Ctrl+C", "Quit Athenas Studio"),
        Line::from(""),
    ];

    let (tab_title, tab_rows): (&str, Vec<(&str, &str)>) = match active_tab {
        0 => (
            " Chat Tab",
            vec![
                ("Enter", "Send message / run slash command"),
                ("Alt+Enter / Ctrl+J", "Insert newline (also Shift+Enter)"),
                ("Alt+↑ / Alt+↓", "Recall prompt history"),
                ("PgUp / PgDn / Wheel", "Scroll conversation"),
                (
                    "Tab / Shift+Tab",
                    "Toggle Thinking (if present) / cycle tabs",
                ),
                ("Esc", "Cancel ongoing generation"),
            ],
        ),
        1 => (
            " Models Tab",
            vec![
                ("↑/↓ or j/k", "Select model"),
                ("Enter", "Load selected model into Chat"),
                ("/", "Filter models live"),
                ("u", "Unload active model (local or server)"),
                ("d / Del", "Delete model (with confirmation)"),
            ],
        ),
        2 => (
            " Browser Tab (HuggingFace)",
            vec![
                ("Enter / →", "Search / inspect repo files / download"),
                ("← / Esc", "Back to results / clear search / Chat"),
                ("Ctrl+G / Ctrl+U", "Toggle GGUF-only filter / clear query"),
                ("/ / r", "Edit search query / reset search"),
                ("Tab / Alt+←/→", "Switch tabs freely even in Search"),
            ],
        ),
        3 => (
            " Server Panel",
            vec![
                ("[ / ] or Ctrl+←/→", "Switch config sub-tabs"),
                ("↑/↓ / PgUp/PgDn", "Navigate fields (±10 on PgUp/PgDn)"),
                ("←/→", "Cycle model, GPU runtime, or toggle ON/OFF"),
                ("Enter / Space", "Edit value / toggle ON-OFF / run action"),
                ("s / l / u / k", "Start-Stop / Load / Unload / API Keys"),
            ],
        ),
        4 => (
            " Settings Tab",
            vec![
                ("↑/↓ or j/k", "Navigate settings (PgUp/PgDn ±5)"),
                ("←/→", "Toggle boolean / cycle Backend, Device, GPU"),
                ("Enter / Space", "Toggle / cycle / edit value"),
                ("Esc", "Cancel edit / return to Chat"),
            ],
        ),
        _ => (
            " Logs Tab",
            vec![
                (
                    "a / e / w / i / d",
                    "Filter: ALL / ERROR / WARN / INFO / DEBUG",
                ),
                ("/", "Live text search"),
                ("z", "Toggle line wrapping"),
                ("↑/↓ / PgUp / PgDn", "Scroll logs"),
                ("End / c", "Jump to bottom / Clear logs"),
            ],
        ),
    };

    lines.push(Line::from(Span::styled(
        tab_title,
        Style::default()
            .fg(theme::PRIMARY)
            .add_modifier(Modifier::BOLD),
    )));
    for (k, desc) in tab_rows {
        lines.push(help_row(k, desc));
    }

    lines.push(Line::from(""));
    let mut footer = Vec::new();
    footer.extend(theme::keycap("Esc / ?", "Close"));
    lines.push(Line::from(footer).alignment(Alignment::Center));

    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

/// Centered Update Modal (`U`, `/update`, or clicking the header update pill).
pub fn render_update_modal(
    f: &mut Frame,
    current_version: &str,
    update_available: Option<&athenas_core::UpdateInfo>,
    checking: bool,
    status_msg: Option<&str>,
) {
    let area = f.area();
    let modal_w = 72u16.min(area.width.saturating_sub(4).max(30));
    let modal_h = 20u16.min(area.height.saturating_sub(2).max(12));
    let x = (area.width.saturating_sub(modal_w)) / 2;
    let y = (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect::new(x, y, modal_w, modal_h);

    f.render_widget(Clear, modal_area);

    let accent = if update_available.is_some() {
        theme::WARNING
    } else {
        theme::PRIMARY
    };
    let title = if update_available.is_some() {
        " ↑ Athenas Studio Update Available "
    } else {
        " ◆ Athenas Studio Updates "
    };
    let block =
        theme::rounded_block_colored(title, accent).style(Style::default().bg(theme::BG_SURFACE));
    let inner = block.inner(modal_area);
    f.render_widget(block, modal_area);

    let max_line_w = inner.width.saturating_sub(4) as usize;
    let mut lines: Vec<Line<'static>> = vec![Line::from("")];

    if checking {
        lines.push(Line::from(vec![
            Span::raw("  Current: "),
            Span::styled(
                format!("v{}", current_version),
                Style::default()
                    .fg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "   ◌ Checking GitHub releases…",
                Style::default().fg(theme::WARNING),
            ),
        ]));
    } else if let Some(info) = update_available {
        lines.push(Line::from(vec![
            Span::raw("  Version: "),
            Span::styled(
                format!("v{}", current_version),
                Style::default().fg(theme::TEXT_DIM),
            ),
            Span::styled("  →  ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(
                format!("v{}", info.latest_version),
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("  Release: "),
            Span::styled(
                info.release_url.clone(),
                Style::default().fg(theme::PRIMARY),
            ),
        ]));
        lines.push(Line::from(""));

        if let Some(ref notes) = info.release_notes {
            lines.push(Line::from(Span::styled(
                "  Release Notes:",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            )));
            for note_line in notes
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .take(6)
            {
                let truncated: String = if note_line.chars().count() > max_line_w {
                    format!(
                        "{}…",
                        note_line
                            .chars()
                            .take(max_line_w.saturating_sub(1))
                            .collect::<String>()
                    )
                } else {
                    note_line.to_string()
                };
                lines.push(Line::from(Span::styled(
                    format!("    {}", truncated),
                    Style::default().fg(theme::TEXT_DIM),
                )));
            }
            lines.push(Line::from(""));
        }

        lines.push(Line::from(vec![
            Span::raw("  Update command: "),
            Span::styled(
                " athenas update ",
                Style::default()
                    .fg(theme::BG_HEADER)
                    .bg(theme::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::raw("  Current: "),
            Span::styled(
                format!("v{}", current_version),
                Style::default()
                    .fg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("   ● ", Style::default().fg(theme::SUCCESS)),
            Span::styled(
                status_msg.unwrap_or("You're up to date!").to_string(),
                Style::default()
                    .fg(theme::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::raw("  Update manually anytime with: "),
            Span::styled(
                " athenas update ",
                Style::default()
                    .fg(theme::PRIMARY)
                    .bg(theme::BG_HIGHLIGHT)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    lines.push(Line::from(""));
    let mut footer = Vec::new();
    footer.extend(theme::keycap("r", "Check Now"));
    footer.extend(theme::keycap("Esc", "Close"));
    lines.push(Line::from(footer).alignment(Alignment::Center));

    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

fn help_row(key: &str, desc: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("{:<20}", key),
            Style::default()
                .fg(theme::PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(desc.to_string(), Style::default().fg(theme::TEXT_DIM)),
    ])
}

// ============================================================================
// CHAT TAB (F1)
// ============================================================================

pub fn render_chat_area(
    f: &mut Frame,
    area: Rect,
    state: &mut ChatState,
    chat_input: &mut tui_textarea::TextArea<'static>,
    is_loading_model: bool,
    loading_spinner: usize,
) {
    let input_height = (chat_input.lines().len().max(1) as u16 + 2).min(8);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);

    render_messages(f, chunks[0], state, is_loading_model, loading_spinner);
    f.render_widget(&*chat_input, chunks[1]);
    render_status_bar(f, chunks[2], state);
}

fn render_messages(
    f: &mut Frame,
    area: Rect,
    state: &mut ChatState,
    is_loading_model: bool,
    loading_spinner: usize,
) {
    // Sleek Welcome Card when conversation is empty and idle
    if state.messages.is_empty() && !state.is_generating && !is_loading_model {
        render_chat_welcome(f, area, state);
        return;
    }

    let content_width = area.width.saturating_sub(5) as usize;
    let reasoning_width = area.width.saturating_sub(7) as usize;

    let mut lines: Vec<Line<'static>> = Vec::new();
    let sep_width = area.width.saturating_sub(6) as usize;
    let separator = format!("  {}", "─".repeat(sep_width));

    let msg_count = state.messages.len();
    for idx in 0..msg_count {
        if idx > 0 {
            lines.push(Line::styled(
                separator.clone(),
                Style::default().fg(theme::BG_HIGHLIGHT),
            ));
        }

        let role = state.messages[idx].role.clone();
        let time_str = state.messages[idx].time_str();
        let reasoning = state.messages[idx].reasoning.clone();
        let reasoning_expanded = state.messages[idx].reasoning_expanded;

        let (role_fg, role_bg, label) = match role.as_str() {
            "user" => (theme::BG_HEADER, theme::SUCCESS, "YOU"),
            "assistant" => (theme::BG_HEADER, theme::PRIMARY, "AI"),
            "error" => (theme::BG_HEADER, theme::ERROR, "ERROR"),
            "system" => (theme::BG_HEADER, theme::WARNING, "SYSTEM"),
            _ => (theme::TEXT, theme::BORDER, "MSG"),
        };

        let mut header_spans = vec![Span::raw("  "), theme::pill(label, role_fg, role_bg)];
        if !time_str.is_empty() {
            header_spans.push(Span::styled(
                format!("  {}", time_str),
                Style::default().fg(theme::TEXT_MUTED),
            ));
        }
        lines.push(Line::from(header_spans));
        lines.push(Line::from(""));

        // Collapsible reasoning block
        if role == "assistant" && !reasoning.is_empty() {
            if reasoning_expanded {
                lines.push(Line::from(vec![
                    Span::styled("  ╭─ ", Style::default().fg(theme::REASONING)),
                    Span::styled(
                        "Thinking",
                        Style::default()
                            .fg(theme::REASONING)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "  (Tab to collapse)",
                        Style::default().fg(theme::TEXT_MUTED),
                    ),
                ]));
                for line in reasoning.lines() {
                    for wrapped in wrap_text(line, reasoning_width) {
                        lines.push(Line::from(vec![
                            Span::styled("  │ ", Style::default().fg(theme::REASONING)),
                            Span::styled(
                                wrapped,
                                Style::default()
                                    .fg(theme::REASONING)
                                    .add_modifier(Modifier::DIM),
                            ),
                        ]));
                    }
                }
                lines.push(Line::styled(
                    "  ╰────────────────────",
                    Style::default().fg(theme::REASONING),
                ));
            } else {
                let preview_len = 56;
                let preview: String = reasoning
                    .replace('\n', " ")
                    .chars()
                    .take(preview_len)
                    .collect();
                let suffix = if reasoning.chars().count() > preview_len {
                    "…"
                } else {
                    ""
                };
                lines.push(Line::from(vec![
                    Span::styled("  ▸ ", Style::default().fg(theme::REASONING)),
                    Span::styled(
                        "Thinking",
                        Style::default()
                            .fg(theme::REASONING)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" {}{}", preview, suffix),
                        Style::default().fg(theme::TEXT_MUTED),
                    ),
                    Span::styled("  [Tab]", Style::default().fg(theme::SECONDARY)),
                ]));
            }
            lines.push(Line::from(""));
        }

        // Content
        if role == "assistant" {
            let md_lines = state.cached_markdown(idx, content_width);
            lines.extend(md_lines);
        } else if role == "user" {
            let content = state.messages[idx].content.clone();
            for line in content.lines() {
                for wrapped in wrap_text(line, content_width) {
                    lines.push(Line::styled(
                        format!("  {}", wrapped),
                        Style::default().fg(theme::TEXT),
                    ));
                }
            }
        } else if role == "error" {
            let content = state.messages[idx].content.clone();
            for line in content.lines() {
                for wrapped in wrap_text(line, content_width) {
                    lines.push(Line::styled(
                        format!("  ✗ {}", wrapped),
                        Style::default().fg(theme::ERROR),
                    ));
                }
            }
        } else {
            let content = state.messages[idx].content.clone();
            for line in content.lines() {
                for wrapped in wrap_text(line, content_width) {
                    lines.push(Line::styled(
                        format!("  {}", wrapped),
                        Style::default().fg(theme::TEXT_DIM),
                    ));
                }
            }
        }
        lines.push(Line::from(""));
    }

    // Active streaming message
    if state.is_generating {
        let elapsed = state
            .generation_start
            .map(|s| s.elapsed().as_secs())
            .unwrap_or(0);

        if !state.messages.is_empty() {
            lines.push(Line::styled(
                separator.clone(),
                Style::default().fg(theme::BG_HIGHLIGHT),
            ));
        }

        let spinner = theme::spinner_frame(loading_spinner);
        lines.push(Line::from(vec![
            Span::raw("  "),
            theme::pill("AI", theme::BG_HEADER, theme::PRIMARY),
            Span::styled(
                format!("  {} generating", spinner),
                Style::default().fg(theme::PRIMARY),
            ),
        ]));
        lines.push(Line::from(""));

        if state.streaming_text.is_empty() && state.streaming_reasoning.is_empty() {
            let wait_msg = if elapsed > 60 {
                format!(
                    "  {} Still waiting… {}s (model may be thinking or overloaded)",
                    spinner, elapsed
                )
            } else if elapsed > 30 {
                format!("  {} Waiting for first token… {}s", spinner, elapsed)
            } else if elapsed > 5 {
                format!("  {} Processing prompt… {}s", spinner, elapsed)
            } else {
                format!("  {} Thinking…", spinner)
            };
            let wait_color = if elapsed > 60 {
                theme::WARNING
            } else {
                theme::PRIMARY
            };
            lines.push(Line::styled(
                wait_msg,
                Style::default()
                    .fg(wait_color)
                    .add_modifier(Modifier::ITALIC),
            ));
        } else {
            if !state.streaming_reasoning.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("  ╭─ ", Style::default().fg(theme::REASONING)),
                    Span::styled(
                        "Thinking",
                        Style::default()
                            .fg(theme::REASONING)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  {} live", spinner),
                        Style::default().fg(theme::TEXT_MUTED),
                    ),
                ]));
                for line in state.streaming_reasoning.lines() {
                    for wrapped in wrap_text(line, reasoning_width) {
                        lines.push(Line::from(vec![
                            Span::styled("  │ ", Style::default().fg(theme::REASONING)),
                            Span::styled(
                                wrapped,
                                Style::default()
                                    .fg(theme::REASONING)
                                    .add_modifier(Modifier::DIM),
                            ),
                        ]));
                    }
                }
                lines.push(Line::styled(
                    "  ╰────────────────────",
                    Style::default().fg(theme::REASONING),
                ));
                lines.push(Line::from(""));
            }
            for line in state.streaming_text.lines() {
                for wrapped in wrap_text(line, content_width) {
                    lines.push(Line::styled(
                        format!("  {}", wrapped),
                        Style::default().fg(theme::TEXT),
                    ));
                }
            }
            if elapsed > 1 {
                let info = if let Some(tps) = state.tokens_per_second {
                    format!("  ⚡ {:.1} tok/s · {}s · Esc to stop", tps, elapsed)
                } else {
                    format!("  ⏱ {}s · Esc to stop", elapsed)
                };
                lines.push(Line::from(""));
                lines.push(Line::styled(
                    info,
                    Style::default()
                        .fg(theme::TEXT_MUTED)
                        .add_modifier(Modifier::ITALIC),
                ));
            }
        }
        lines.push(Line::from(""));
    }

    if is_loading_model {
        let spinner = theme::spinner_frame(loading_spinner);
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {} ", spinner),
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Loading model into memory…",
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    let inner_height = area.height.saturating_sub(2) as usize;
    let total_lines = lines.len();
    let max_scroll = total_lines.saturating_sub(inner_height);
    state.max_scroll = max_scroll;

    if !state.auto_scroll && state.scroll >= max_scroll {
        state.auto_scroll = true;
    }

    let scroll = if state.auto_scroll {
        max_scroll
    } else {
        state.scroll.min(max_scroll)
    };

    let title = if total_lines > inner_height && !state.auto_scroll {
        let pct = if max_scroll > 0 {
            ((scroll as f32 / max_scroll as f32) * 100.0) as u32
        } else {
            0
        };
        format!(" ✦ Conversation [{}%] ", pct)
    } else {
        " ✦ Conversation ".to_string()
    };

    let block = theme::rounded_block(title, true);
    let paragraph = Paragraph::new(lines)
        .block(block)
        .scroll((scroll.min(u16::MAX as usize) as u16, 0));

    f.render_widget(paragraph, area);
    if total_lines > inner_height {
        theme::render_scrollbar(f, area, max_scroll + 1, scroll);
    }
}

fn render_chat_welcome(f: &mut Frame, area: Rect, state: &ChatState) {
    let block = theme::rounded_block(" ✦ Athenas Studio ", true);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let model_line = if let Some(ref m) = state.current_model {
        Line::from(vec![
            Span::styled(
                "● Ready  ",
                Style::default()
                    .fg(theme::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                short_model_name(m),
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                state
                    .current_backend
                    .as_deref()
                    .map(|b| format!("  [{}]", b))
                    .unwrap_or_default(),
                Style::default().fg(theme::SECONDARY),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("○ No model loaded  ", Style::default().fg(theme::WARNING)),
            Span::styled("Press ", Style::default().fg(theme::TEXT_DIM)),
            Span::styled(
                " F2 ",
                Style::default()
                    .fg(theme::PRIMARY)
                    .bg(theme::BG_HIGHLIGHT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " to pick a local model or ",
                Style::default().fg(theme::TEXT_DIM),
            ),
            Span::styled(
                " F3 ",
                Style::default()
                    .fg(theme::PRIMARY)
                    .bg(theme::BG_HIGHLIGHT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" to download one", Style::default().fg(theme::TEXT_DIM)),
        ])
    };

    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "◆  A T H E N A S   S T U D I O",
            Style::default()
                .fg(theme::PRIMARY)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Local LLM Inference & OpenAI-Compatible API Workstation",
            Style::default().fg(theme::TEXT_MUTED),
        )),
        Line::from(""),
        model_line,
        Line::from(""),
        Line::from(Span::styled(
            "──────────────────────────────────────────────────────",
            Style::default().fg(theme::BG_HIGHLIGHT),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "/model ",
                Style::default()
                    .fg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("switch model   ", Style::default().fg(theme::TEXT_DIM)),
            Span::styled(
                "/system ",
                Style::default()
                    .fg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("set prompt   ", Style::default().fg(theme::TEXT_DIM)),
            Span::styled(
                "/clear ",
                Style::default()
                    .fg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("reset   ", Style::default().fg(theme::TEXT_DIM)),
            Span::styled(
                "/help ",
                Style::default()
                    .fg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("all cmds", Style::default().fg(theme::TEXT_DIM)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Alt+Enter / Ctrl+J ", Style::default().fg(theme::SECONDARY)),
            Span::styled("multiline   ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled("Alt+↑/↓ ", Style::default().fg(theme::SECONDARY)),
            Span::styled("history   ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled("Tab ", Style::default().fg(theme::SECONDARY)),
            Span::styled("thinking   ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled("? ", Style::default().fg(theme::SECONDARY)),
            Span::styled("shortcuts", Style::default().fg(theme::TEXT_MUTED)),
        ]),
    ];

    // Vertically center the welcome card if there's room
    let top_pad = inner.height.saturating_sub(lines.len() as u16) / 2;
    if top_pad > 0 {
        let mut padded = vec![Line::from(""); top_pad as usize];
        padded.append(&mut lines);
        lines = padded;
    }

    let p = Paragraph::new(lines).alignment(Alignment::Center);
    f.render_widget(p, inner);
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut result: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_len: usize = 0;

    for word in text.split(' ') {
        let word_len = word.chars().count();

        if current.is_empty() {
            if word_len <= width {
                current = word.to_string();
                current_len = word_len;
            } else {
                let mut chars: Vec<char> = word.chars().collect();
                while chars.len() > width {
                    let chunk: String = chars.drain(..width).collect();
                    result.push(chunk);
                }
                if !chars.is_empty() {
                    current = chars.iter().collect();
                    current_len = chars.len();
                }
            }
        } else if current_len + 1 + word_len <= width {
            current.push(' ');
            current.push_str(word);
            current_len += 1 + word_len;
        } else {
            result.push(std::mem::take(&mut current));
            current_len = 0;

            if word_len <= width {
                current = word.to_string();
                current_len = word_len;
            } else {
                let mut chars: Vec<char> = word.chars().collect();
                while chars.len() > width {
                    let chunk: String = chars.drain(..width).collect();
                    result.push(chunk);
                }
                if !chars.is_empty() {
                    current = chars.iter().collect();
                    current_len = chars.len();
                }
            }
        }
    }

    if !current.is_empty() || result.is_empty() {
        result.push(current);
    }

    result
}

pub fn short_model_name(path: &str) -> String {
    let filename = path.rsplit(['/', '\\']).next().unwrap_or(path);
    filename
        .strip_suffix(".gguf")
        .or_else(|| filename.strip_suffix(".safetensors"))
        .unwrap_or(filename)
        .to_string()
}

/// Match whether a registry `ModelInfo` corresponds to the currently loaded model string.
pub fn is_model_loaded(model: &ModelInfo, loaded_model: Option<&str>) -> bool {
    let Some(loaded) = loaded_model else {
        return false;
    };
    if loaded.is_empty() {
        return false;
    }
    if model.name == loaded || model.id == loaded {
        return true;
    }
    let path_str = model.file_path.to_string_lossy();
    if path_str == loaded {
        return true;
    }
    let file_name = model
        .file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let file_stem = model
        .file_path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let loaded_short = short_model_name(loaded);
    let model_short = short_model_name(&model.name);
    file_name == loaded
        || file_stem == loaded
        || file_stem == loaded_short
        || model_short == loaded_short
}

fn render_status_bar(f: &mut Frame, area: Rect, state: &ChatState) {
    let sep = Span::styled(" │ ", Style::default().fg(theme::BORDER));
    let mut parts: Vec<Span> = Vec::new();

    if let Some(ref model) = state.current_model {
        parts.push(Span::styled(" ◆ ", Style::default().fg(theme::PRIMARY)));
        parts.push(Span::styled(
            short_model_name(model),
            Style::default()
                .fg(theme::PRIMARY)
                .add_modifier(Modifier::BOLD),
        ));
    } else {
        parts.push(Span::styled(
            " ○ No model",
            Style::default().fg(theme::WARNING),
        ));
    }

    if let Some(ref backend) = state.current_backend {
        parts.push(sep.clone());
        parts.push(Span::styled(
            backend.clone(),
            Style::default().fg(theme::SECONDARY),
        ));
    }

    if !state.gpu_info.is_empty() {
        let layers_str = if state.gpu_layers < 0 {
            "all".to_string()
        } else if state.gpu_layers == 0 {
            "CPU".to_string()
        } else {
            format!("{}L", state.gpu_layers)
        };
        parts.push(sep.clone());
        parts.push(Span::styled(
            format!("{} [{}·{}]", state.gpu_info, state.gpu_runtime, layers_str),
            Style::default().fg(theme::REASONING),
        ));
    } else if state.gpu_layers == 0 {
        parts.push(sep.clone());
        parts.push(Span::styled("CPU", Style::default().fg(theme::WARNING)));
    }

    if let Some(tps) = state.tokens_per_second {
        parts.push(sep.clone());
        parts.push(Span::styled(
            format!("⚡ {:.1} tok/s", tps),
            Style::default().fg(theme::SUCCESS),
        ));
    }

    if !state.system_prompt.is_empty() {
        parts.push(sep.clone());
        parts.push(Span::styled("✦ sys", Style::default().fg(theme::REASONING)));
    }

    let left_len: usize = parts.iter().map(|s| s.content.chars().count()).sum();
    let available = area.width as usize;

    let shortcuts_full =
        "Enter: Send · Alt+Enter: Newline · Alt+←/→: Tabs · Tab: Think/Next · Esc: Stop ";
    let shortcuts_mid = "Enter: Send · Alt+←/→: Tabs · Tab · Esc · /help ";
    let shortcuts_min = "Enter · Esc · /help ";

    let remaining = available.saturating_sub(left_len + 2);
    let shortcuts = if remaining >= shortcuts_full.chars().count() {
        shortcuts_full
    } else if remaining >= shortcuts_mid.chars().count() {
        shortcuts_mid
    } else {
        shortcuts_min
    };
    let sc_len = shortcuts.chars().count();
    let padding = available.saturating_sub(left_len + sc_len);
    parts.push(Span::raw(" ".repeat(padding)));
    parts.push(Span::styled(
        shortcuts,
        Style::default().fg(theme::TEXT_MUTED),
    ));

    let p = Paragraph::new(Line::from(parts)).style(Style::default().bg(theme::BG_HEADER));
    f.render_widget(p, area);
}

// ============================================================================
// MODELS TAB (F2) — 2-Column Split + Inspector Card + Filter + Confirm Modal
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub fn render_model_list(
    f: &mut Frame,
    area: Rect,
    state: &ModelListState,
    loaded_model: Option<&str>,
    loaded_backend: Option<&str>,
    hardware: &HardwareInfo,
    context_size: u32,
    gpu_layers: i32,
) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    let (list_area, inspector_area) = if outer[0].width >= 76 {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
            .split(outer[0]);
        (cols[0], Some(cols[1]))
    } else {
        (outer[0], None)
    };

    // Left column: optional filter bar + model list
    let list_chunks = if state.filtering || !state.filter.is_empty() {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(3)])
            .split(list_area)
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(0), Constraint::Min(3)])
            .split(list_area)
    };

    if state.filtering || !state.filter.is_empty() {
        let cursor = if state.filtering { "▏" } else { "" };
        let filter_block = theme::rounded_block(" Filter (/) ", state.filtering);
        let filter_p = Paragraph::new(Line::from(vec![
            Span::styled(" / ", Style::default().fg(theme::PRIMARY)),
            Span::styled(
                format!("{}{}", state.filter, cursor),
                Style::default().fg(theme::TEXT),
            ),
        ]))
        .block(filter_block);
        f.render_widget(filter_p, list_chunks[0]);
    }

    let filtered = state.filtered_models();
    let title = format!(" Local Models ({}/{}) ", filtered.len(), state.models.len());
    let list_block = theme::rounded_block(title, !state.filtering);

    if state.models.is_empty() {
        let empty = Paragraph::new(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No local models found.",
                Style::default().fg(theme::TEXT_DIM),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Press ", Style::default().fg(theme::TEXT_MUTED)),
                Span::styled(
                    " F3 ",
                    Style::default()
                        .fg(theme::PRIMARY)
                        .bg(theme::BG_HIGHLIGHT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    " to browse & download from HuggingFace, or run ",
                    Style::default().fg(theme::TEXT_MUTED),
                ),
                Span::styled(
                    "athenas models pull <repo>",
                    Style::default().fg(theme::PRIMARY),
                ),
            ]),
        ])
        .block(list_block);
        f.render_widget(empty, list_chunks[1]);
    } else if filtered.is_empty() {
        let empty = Paragraph::new(Line::styled(
            "  No models match the current filter. Press Esc to clear.",
            Style::default().fg(theme::TEXT_MUTED),
        ))
        .block(list_block);
        f.render_widget(empty, list_chunks[1]);
    } else {
        let items: Vec<ListItem> = filtered
            .iter()
            .map(|m| {
                let is_loaded = is_model_loaded(m, loaded_model);
                let (cat_tag, cat_color) = model_category_badge(m);

                let mut spans = vec![
                    Span::styled(
                        format!("[{}] ", cat_tag),
                        Style::default().fg(cat_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        m.name.clone(),
                        Style::default()
                            .fg(if is_loaded {
                                theme::SUCCESS
                            } else {
                                theme::TEXT
                            })
                            .add_modifier(if is_loaded {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    ),
                ];

                if is_loaded {
                    spans.push(Span::styled(
                        " ● LOADED",
                        Style::default()
                            .fg(theme::SUCCESS)
                            .add_modifier(Modifier::BOLD),
                    ));
                }

                if let Some(ref q) = m.quantization {
                    spans.push(Span::styled(
                        format!(" [{}]", q),
                        Style::default().fg(theme::WARNING),
                    ));
                }
                spans.push(Span::styled(
                    format!("  {}", format_bytes_binary(m.file_size_bytes)),
                    Style::default().fg(theme::TEXT_MUTED),
                ));
                ListItem::new(Line::from(spans))
            })
            .collect();

        let list = List::new(items)
            .block(list_block)
            .highlight_style(
                Style::default()
                    .bg(theme::BG_HIGHLIGHT)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▶ ");

        let mut list_state = state.list_state.clone();
        f.render_stateful_widget(list, list_chunks[1], &mut list_state);
        theme::render_scrollbar(
            f,
            list_chunks[1],
            filtered.len(),
            state.list_state.selected().unwrap_or(0),
        );
    }

    // Right column: Model Inspector Card
    if let Some(insp_area) = inspector_area {
        render_model_inspector(
            f,
            insp_area,
            state.selected(),
            loaded_model,
            loaded_backend,
            hardware,
            context_size,
            gpu_layers,
        );
    }

    // Bottom status/hints bar
    if let Some(ref msg) = state.status_message {
        let color = if state.status_is_error {
            theme::ERROR
        } else {
            theme::SUCCESS
        };
        let p = Paragraph::new(Line::styled(
            format!(" {} ", msg),
            Style::default().fg(color),
        ))
        .style(Style::default().bg(theme::BG_HEADER))
        .alignment(Alignment::Center);
        f.render_widget(p, outer[1]);
    } else {
        let mut hints = Vec::new();
        hints.extend(theme::keycap("Enter", "Load"));
        hints.extend(theme::keycap("/", "Filter"));
        hints.extend(theme::keycap("u", "Unload"));
        hints.extend(theme::keycap("d", "Delete"));
        hints.extend(theme::keycap("?", "Help"));
        let p = Paragraph::new(Line::from(hints))
            .style(Style::default().bg(theme::BG_HEADER))
            .alignment(Alignment::Center);
        f.render_widget(p, outer[1]);
    }

    // Delete confirmation modal overlay
    if let Some((ref name, ref size)) = state.confirm_delete {
        render_delete_confirm_modal(f, name, size);
    }
}

fn model_category_badge(m: &ModelInfo) -> (&'static str, Color) {
    let cat = m.category.as_deref().unwrap_or("").to_lowercase();
    let arch = m.architecture.as_deref().unwrap_or("").to_lowercase();
    if cat.contains("whisper") || arch.contains("whisper") {
        ("WSP", theme::REASONING)
    } else if cat.contains("embed") || arch.contains("embed") || arch.contains("bert") {
        ("EMB", theme::SECONDARY)
    } else if cat.contains("clef") || arch.contains("clef") {
        ("CLF", theme::WARNING)
    } else {
        ("LLM", theme::PRIMARY)
    }
}

#[allow(clippy::too_many_arguments)]
fn render_model_inspector(
    f: &mut Frame,
    area: Rect,
    selected: Option<&ModelInfo>,
    loaded_model: Option<&str>,
    loaded_backend: Option<&str>,
    hardware: &HardwareInfo,
    context_size: u32,
    gpu_layers: i32,
) {
    let block = theme::rounded_block(" ◆ Model Inspector ", false);
    let Some(m) = selected else {
        let p = Paragraph::new(Line::styled(
            "  Select a model to inspect details & memory fit.",
            Style::default().fg(theme::TEXT_MUTED),
        ))
        .block(block);
        f.render_widget(p, area);
        return;
    };

    let is_loaded = is_model_loaded(m, loaded_model);
    let (cat_tag, cat_color) = model_category_badge(m);

    let size_mb = m.file_size_bytes / (1024 * 1024);
    let effective_gpu_layers =
        if m.format == athenas_core::ModelFormat::Onnx && !athenas_inference::GPU_OFFLOAD_CAPABLE {
            0
        } else {
            gpu_layers
        };
    let est =
        athenas_core::estimate_model_memory(size_mb, context_size, effective_gpu_layers, hardware);
    let fits = est.fits(hardware);
    let tight = fits
        && hardware.memory_available_mb > 0
        && est.ram_mb * 10 >= hardware.memory_available_mb * 9;

    let (verdict_text, verdict_color) = if !fits {
        ("✗ OOM", theme::ERROR)
    } else if tight {
        ("⚠ TIGHT", theme::WARNING)
    } else {
        ("✓ FITS", theme::SUCCESS)
    };

    let mut lines = vec![
        Line::from(vec![
            Span::raw(" "),
            theme::pill(cat_tag, theme::BG_HEADER, cat_color),
            Span::raw(" "),
            if is_loaded {
                theme::pill("● LOADED", theme::BG_HEADER, theme::SUCCESS)
            } else {
                theme::pill(verdict_text, theme::BG_HEADER, verdict_color)
            },
        ]),
        Line::from(""),
        inspector_kv("Name", &m.name, theme::TEXT),
        inspector_kv("Repo", &m.repo_id, theme::TEXT_DIM),
        inspector_kv("Format", &format!("{:?}", m.format), theme::SECONDARY),
        inspector_kv(
            "Quant",
            m.quantization.as_deref().unwrap_or("—"),
            theme::WARNING,
        ),
        inspector_kv(
            "Arch",
            m.architecture.as_deref().unwrap_or("—"),
            theme::PRIMARY,
        ),
        inspector_kv(
            "Context",
            &m.context_length
                .map(|c| format!("{} tokens", c))
                .unwrap_or_else(|| "—".to_string()),
            theme::TEXT,
        ),
        inspector_kv("Size", &format_bytes_binary(m.file_size_bytes), theme::TEXT),
        Line::from(""),
        Line::from(Span::styled(
            " ── Memory Estimate ──",
            Style::default()
                .fg(theme::PRIMARY)
                .add_modifier(Modifier::BOLD),
        )),
        inspector_kv("RAM est", &format!("~{}", fmt_mb(est.ram_mb)), theme::TEXT),
    ];

    if let Some(vram) = est.vram_mb {
        lines.push(inspector_kv(
            "VRAM est",
            &format!("~{}", fmt_mb(vram)),
            theme::REASONING,
        ));
    }
    lines.push(inspector_kv(
        "RAM free",
        &fmt_mb(hardware.memory_available_mb),
        theme::TEXT_DIM,
    ));
    lines.push(Line::from(vec![
        Span::styled(" Fit:      ", Style::default().fg(theme::TEXT_MUTED)),
        Span::styled(
            verdict_text,
            Style::default()
                .fg(verdict_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  (ctx {})", context_size),
            Style::default().fg(theme::TEXT_MUTED),
        ),
    ]));

    if is_loaded {
        if let Some(b) = loaded_backend {
            lines.push(inspector_kv("Backend", b, theme::SUCCESS));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " Path:",
        Style::default().fg(theme::TEXT_MUTED),
    )));
    lines.push(Line::from(Span::styled(
        format!(" {}", m.file_path.display()),
        Style::default().fg(theme::TEXT_DIM),
    )));

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn inspector_kv(label: &str, value: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!(" {:<9} ", format!("{}:", label)),
            Style::default().fg(theme::TEXT_MUTED),
        ),
        Span::styled(value.to_string(), Style::default().fg(color)),
    ])
}

fn render_delete_confirm_modal(f: &mut Frame, name: &str, size: &str) {
    let area = f.area();
    let w = 56u16.min(area.width.saturating_sub(4).max(24));
    let h = 9u16.min(area.height.saturating_sub(2).max(7));
    let x = (area.width.saturating_sub(w)) / 2;
    let y = (area.height.saturating_sub(h)) / 2;
    let modal = Rect::new(x, y, w, h);

    f.render_widget(Clear, modal);
    let block = theme::rounded_block_colored(" ⚠ Confirm Delete Model ", theme::ERROR)
        .style(Style::default().bg(theme::BG_SURFACE));
    let inner = block.inner(modal);
    f.render_widget(block, modal);

    let mut action_spans = Vec::new();
    action_spans.extend(theme::keycap("y", "Delete permanently"));
    action_spans.extend(theme::keycap("n / Esc", "Cancel"));

    let p = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Delete ", Style::default().fg(theme::TEXT)),
            Span::styled(
                name,
                Style::default()
                    .fg(theme::WARNING)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" ({})?", size),
                Style::default().fg(theme::TEXT_DIM),
            ),
        ]),
        Line::from(Span::styled(
            "This cannot be undone.",
            Style::default().fg(theme::ERROR),
        )),
        Line::from(""),
        Line::from(action_spans),
    ])
    .alignment(Alignment::Center)
    .wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

// ============================================================================
// BROWSER TAB (F3) — Async Spinner, Proper Scrolling, Dedicated Download Card
// ============================================================================

pub fn render_model_browser(
    f: &mut Frame,
    area: Rect,
    state: &ModelBrowserState,
    spinner_tick: usize,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    let block = theme::rounded_block(" ◆ HuggingFace Model Browser ", true);
    let inner = block.inner(chunks[0]);
    f.render_widget(block, chunks[0]);

    match &state.phase {
        BrowserPhase::Search => {
            let mut lines = vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  Search HuggingFace Hub",
                    Style::default()
                        .fg(theme::PRIMARY)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled(
                        "  ❯ ",
                        Style::default()
                            .fg(theme::PRIMARY)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("{}▏", state.search_input),
                        Style::default()
                            .fg(theme::TEXT)
                            .bg(theme::BG_HIGHLIGHT)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  Filter: ", Style::default().fg(theme::TEXT_MUTED)),
                    if state.gguf_only {
                        theme::pill("GGUF ONLY", theme::BG_HEADER, theme::SUCCESS)
                    } else {
                        theme::pill("ALL FORMATS", theme::TEXT, theme::BG_HIGHLIGHT)
                    },
                    Span::styled(
                        "   (Ctrl+G to toggle · Ctrl+U to clear)",
                        Style::default().fg(theme::TEXT_MUTED),
                    ),
                ]),
            ];

            if state.is_busy {
                let sp = theme::spinner_frame(spinner_tick);
                let lbl = state.busy_label.as_deref().unwrap_or("Searching…");
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {} ", sp),
                        Style::default()
                            .fg(theme::PRIMARY)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(lbl, Style::default().fg(theme::PRIMARY)),
                ]));
            } else if let Some(ref msg) = state.status_message {
                lines.push(Line::from(""));
                lines.push(browser_status_line(msg, state.status_is_error));
            }

            let p = Paragraph::new(lines).wrap(Wrap { trim: false });
            f.render_widget(p, inner);
        }

        BrowserPhase::Results => {
            let sub = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(2),
                    Constraint::Min(3),
                    Constraint::Length(if state.is_busy || state.status_message.is_some() {
                        2
                    } else {
                        0
                    }),
                ])
                .split(inner);

            let header = Paragraph::new(Line::from(vec![
                Span::styled("  Results for ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    format!("'{}'", state.search_input),
                    Style::default()
                        .fg(theme::PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  ({} repos)", state.search_results.len()),
                    Style::default().fg(theme::TEXT_MUTED),
                ),
            ]));
            f.render_widget(header, sub[0]);

            if state.search_results.is_empty() {
                let empty = Paragraph::new(Line::styled(
                    "  No models found. Press / to edit your search.",
                    Style::default().fg(theme::TEXT_MUTED),
                ));
                f.render_widget(empty, sub[1]);
            } else {
                let mut lines: Vec<Line> = Vec::with_capacity(state.search_results.len());
                for (i, result) in state.search_results.iter().enumerate() {
                    let is_selected = i == state.results_selected;
                    let prefix = if is_selected { " ▶ " } else { "   " };
                    let dl = if result.downloads >= 1_000_000 {
                        format!("{:.1}M", result.downloads as f64 / 1_000_000.0)
                    } else if result.downloads >= 1000 {
                        format!("{:.1}k", result.downloads as f64 / 1000.0)
                    } else {
                        result.downloads.to_string()
                    };

                    let row_bg = if is_selected {
                        theme::BG_HIGHLIGHT
                    } else {
                        Color::Reset
                    };
                    let id_style = if is_selected {
                        Style::default()
                            .fg(theme::TEXT)
                            .bg(row_bg)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(theme::TEXT_DIM)
                    };

                    lines.push(Line::from(vec![
                        Span::styled(prefix, Style::default().fg(theme::PRIMARY).bg(row_bg)),
                        Span::styled(result.id.clone(), id_style),
                        Span::styled(
                            format!("  ↓ {}  ♥ {}", dl, result.likes),
                            Style::default().fg(theme::TEXT_MUTED).bg(row_bg),
                        ),
                    ]));
                }

                let visible_h = sub[1].height as usize;
                let scroll = if visible_h > 0 && state.results_selected >= visible_h {
                    (state.results_selected - visible_h + 1) as u16
                } else {
                    0
                };
                let list_p = Paragraph::new(lines).scroll((scroll, 0));
                f.render_widget(list_p, sub[1]);
                if state.search_results.len() > visible_h {
                    theme::render_scrollbar(
                        f,
                        sub[1],
                        state.search_results.len(),
                        state.results_selected,
                    );
                }
            }

            if state.is_busy {
                let sp = theme::spinner_frame(spinner_tick);
                let lbl = state.busy_label.as_deref().unwrap_or("Loading files…");
                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(format!("  {} ", sp), Style::default().fg(theme::PRIMARY)),
                        Span::styled(lbl, Style::default().fg(theme::PRIMARY)),
                    ])),
                    sub[2],
                );
            } else if let Some(ref msg) = state.status_message {
                f.render_widget(
                    Paragraph::new(browser_status_line(msg, state.status_is_error)),
                    sub[2],
                );
            }
        }

        BrowserPhase::SelectFile => {
            let sub = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(2), Constraint::Min(3)])
                .split(inner);

            let repo_name = state
                .selected_result()
                .map(|r| r.id.as_str())
                .unwrap_or("Repository");
            let header = Paragraph::new(Line::from(vec![
                Span::styled("  Select file from ", Style::default().fg(theme::TEXT_DIM)),
                Span::styled(
                    repo_name,
                    Style::default()
                        .fg(theme::PRIMARY)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  ({} files)", state.file_options.len()),
                    Style::default().fg(theme::TEXT_MUTED),
                ),
            ]));
            f.render_widget(header, sub[0]);

            let mut lines: Vec<Line> = Vec::with_capacity(state.file_options.len());
            for (i, (name, size, _sha)) in state.file_options.iter().enumerate() {
                let is_selected = i == state.file_selected;
                let prefix = if is_selected { " ▶ " } else { "   " };
                let row_bg = if is_selected {
                    theme::BG_HIGHLIGHT
                } else {
                    Color::Reset
                };

                let size_str = size
                    .map(format_bytes_binary)
                    .unwrap_or_else(|| "?".to_string());

                let label = name
                    .strip_prefix("onnx-dir:")
                    .map(|d| {
                        if d.is_empty() {
                            "[ONNX] (root)".to_string()
                        } else {
                            format!("[ONNX] {}", d)
                        }
                    })
                    .unwrap_or_else(|| name.clone());

                let rec = is_recommended_quant(&label);
                let mut spans = vec![
                    Span::styled(prefix, Style::default().fg(theme::PRIMARY).bg(row_bg)),
                    Span::styled(
                        label,
                        Style::default()
                            .fg(if is_selected {
                                theme::TEXT
                            } else {
                                theme::TEXT_DIM
                            })
                            .bg(row_bg)
                            .add_modifier(if is_selected {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    ),
                    Span::styled(
                        format!("  ({})", size_str),
                        Style::default().fg(theme::TEXT_MUTED).bg(row_bg),
                    ),
                ];
                if rec {
                    spans.push(Span::styled(
                        "  ★ rec",
                        Style::default()
                            .fg(theme::SUCCESS)
                            .bg(row_bg)
                            .add_modifier(Modifier::BOLD),
                    ));
                }
                lines.push(Line::from(spans));
            }

            let visible_h = sub[1].height as usize;
            let scroll = if visible_h > 0 && state.file_selected >= visible_h {
                (state.file_selected - visible_h + 1) as u16
            } else {
                0
            };
            let list_p = Paragraph::new(lines).scroll((scroll, 0));
            f.render_widget(list_p, sub[1]);
            if state.file_options.len() > visible_h {
                theme::render_scrollbar(f, sub[1], state.file_options.len(), state.file_selected);
            }
        }

        BrowserPhase::Downloading => {
            // Dedicated Download Progress Card with proper layout chunks (zero overdraw!)
            let dl_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1), // Spacer
                    Constraint::Length(3), // Filename & status
                    Constraint::Length(3), // Styled Gauge (own chunk!)
                    Constraint::Length(2), // Speed & ETA
                    Constraint::Min(1),    // Hint
                ])
                .split(inner);

            let sp = theme::spinner_frame(spinner_tick);
            let filename = state.download_filename.as_deref().unwrap_or("Downloading…");

            let info = Paragraph::new(vec![
                Line::from(vec![Span::styled(
                    format!("  {} Downloading Model", sp),
                    Style::default()
                        .fg(theme::PRIMARY)
                        .add_modifier(Modifier::BOLD),
                )]),
                Line::from(vec![
                    Span::styled("  File: ", Style::default().fg(theme::TEXT_MUTED)),
                    Span::styled(filename, Style::default().fg(theme::TEXT)),
                ]),
            ]);
            f.render_widget(info, dl_chunks[1]);

            let (percent, ratio_label, speed_eta_line) =
                if let Some((downloaded, total, speed_mbps)) = state.download_progress {
                    let pct = if total > 0 {
                        ((downloaded as f64 / total as f64) * 100.0).clamp(0.0, 100.0) as u16
                    } else {
                        0
                    };
                    let dl_str = format_bytes_binary(downloaded);
                    let total_str = if total > 0 {
                        format_bytes_binary(total)
                    } else {
                        "?".to_string()
                    };
                    let eta_str = if total > downloaded && speed_mbps > 0.01 {
                        let rem_mb = (total - downloaded) as f64 / 1_048_576.0;
                        let secs = (rem_mb / speed_mbps) as u64;
                        if secs >= 60 {
                            format!("{}m {:02}s", secs / 60, secs % 60)
                        } else {
                            format!("{}s", secs)
                        }
                    } else {
                        "—".to_string()
                    };
                    (
                        pct,
                        format!("{} / {} ({}%)", dl_str, total_str, pct),
                        Line::from(vec![
                            Span::styled("  Speed: ", Style::default().fg(theme::TEXT_MUTED)),
                            Span::styled(
                                format!("{:.2} MB/s", speed_mbps),
                                Style::default()
                                    .fg(theme::SUCCESS)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled("   ETA: ", Style::default().fg(theme::TEXT_MUTED)),
                            Span::styled(eta_str, Style::default().fg(theme::PRIMARY)),
                        ]),
                    )
                } else {
                    (
                        0,
                        "Connecting…".to_string(),
                        Line::styled(
                            "  Connecting to HuggingFace Hub…",
                            Style::default().fg(theme::TEXT_DIM),
                        ),
                    )
                };

            let gauge_margin = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(2),
                    Constraint::Min(10),
                    Constraint::Length(2),
                ])
                .split(dl_chunks[2]);

            let gauge = Gauge::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(theme::BORDER)),
                )
                .gauge_style(
                    Style::default()
                        .fg(theme::PRIMARY)
                        .bg(theme::BG_HIGHLIGHT)
                        .add_modifier(Modifier::BOLD),
                )
                .percent(percent)
                .label(ratio_label);
            f.render_widget(gauge, gauge_margin[1]);

            f.render_widget(Paragraph::new(speed_eta_line), dl_chunks[3]);
            f.render_widget(
                Paragraph::new(Line::styled(
                    "  Press Esc to cancel download",
                    Style::default().fg(theme::TEXT_MUTED),
                )),
                dl_chunks[4],
            );
        }
    }

    // Bottom bar
    let mut hints = Vec::new();
    match &state.phase {
        BrowserPhase::Search => {
            hints.extend(theme::keycap("Enter", "Search"));
            hints.extend(theme::keycap("Ctrl+G", "GGUF filter"));
            hints.extend(theme::keycap("Ctrl+U", "Clear"));
            hints.extend(theme::keycap("Tab/Alt+←/→", "Tabs"));
            hints.extend(theme::keycap("Esc", "Clear/Back"));
        }
        BrowserPhase::Results => {
            hints.extend(theme::keycap("↑/↓", "Navigate"));
            hints.extend(theme::keycap("Enter/→", "Inspect Files"));
            hints.extend(theme::keycap("/", "Edit Query"));
            hints.extend(theme::keycap("r", "Reset"));
        }
        BrowserPhase::SelectFile => {
            hints.extend(theme::keycap("↑/↓", "Select"));
            hints.extend(theme::keycap("Enter", "Download"));
            hints.extend(theme::keycap("←/Esc", "Back"));
        }
        BrowserPhase::Downloading => {
            hints.extend(theme::keycap("Esc", "Cancel Download"));
        }
    }
    let bar = Paragraph::new(Line::from(hints))
        .style(Style::default().bg(theme::BG_HEADER))
        .alignment(Alignment::Center);
    f.render_widget(bar, chunks[1]);
}

fn browser_status_line(msg: &str, is_error: bool) -> Line<'static> {
    let (prefix, color) = if is_error {
        ("✗", theme::ERROR)
    } else {
        ("✓", theme::SUCCESS)
    };
    Line::from(vec![
        Span::styled(
            format!("  {} ", prefix),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(msg.to_string(), Style::default().fg(color)),
    ])
}

// ============================================================================
// SERVER PANEL (F4) — Pinned Telemetry + Quick Actions + Sub-Tabs
// ============================================================================

pub fn render_server_panel(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let has_loaded = state.phase == ServerPhase::Running && !state.loaded_models.is_empty();
    let loaded_h = if has_loaded {
        (state.loaded_models.len() as u16 + 2).min(6)
    } else {
        0
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),        // Pinned Control & Telemetry Banner
            Constraint::Length(1),        // Sub-tab bar
            Constraint::Min(6),           // Scrollable config fields for active sub-tab
            Constraint::Length(loaded_h), // Compact loaded models strip (when running)
            Constraint::Length(1),        // Status bar
        ])
        .split(area);

    render_server_telemetry_banner(f, chunks[0], state);
    render_server_subtab_bar(f, chunks[1], state.active_subtab);
    render_config_fields(f, chunks[2], state);
    if loaded_h > 0 {
        render_loaded_models_panel(f, chunks[3], state);
    }
    render_server_status_bar(f, chunks[4], state);
}

fn render_server_telemetry_banner(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let block = theme::rounded_block(" ◆ Server Control & Telemetry ", true);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Status pill + URL + quick actions
            Constraint::Length(1), // RAM gauge + VRAM gauge
            Constraint::Length(1), // Est. load verdict
            Constraint::Length(1), // Pinned quick-action bar
            Constraint::Min(0),
        ])
        .split(inner);

    // Row 0: Server status pill + URL + Active model
    let status_pill = match state.phase {
        ServerPhase::Running => theme::pill("● RUNNING", theme::BG_HEADER, theme::SUCCESS),
        ServerPhase::LoadingModel => theme::pill("⟳ LOADING", theme::BG_HEADER, theme::PRIMARY),
        ServerPhase::Error => theme::pill("✗ ERROR", theme::BG_HEADER, theme::ERROR),
        ServerPhase::Configuring => theme::pill("○ STOPPED", theme::TEXT_DIM, theme::BG_HIGHLIGHT),
    };
    let mut row0 = vec![Span::raw(" "), status_pill];
    if let Some(ref url) = state.server_url {
        row0.push(Span::styled(
            format!("  {}", url),
            Style::default().fg(theme::PRIMARY),
        ));
    } else {
        row0.push(Span::styled(
            format!("  http://{}:{}", state.host, state.port),
            Style::default().fg(theme::TEXT_MUTED),
        ));
    }
    if let Some(ref m) = state.loaded_model_name {
        row0.push(Span::styled("  │  ", Style::default().fg(theme::BORDER)));
        row0.push(Span::styled(
            short_model_name(m),
            Style::default()
                .fg(theme::TEXT)
                .add_modifier(Modifier::BOLD),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(row0)), rows[0]);

    // Row 1: RAM & VRAM visual Gauges side-by-side
    let gauge_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(1),
            Constraint::Percentage(48),
            Constraint::Length(2),
            Constraint::Percentage(48),
            Constraint::Length(1),
        ])
        .split(rows[1]);

    let ram_total = state.hardware.memory_total_mb;
    let ram_avail = state.hardware.memory_available_mb;
    let ram_used = ram_total.saturating_sub(ram_avail);
    let ram_pct = if ram_total > 0 {
        ((ram_used as f64 / ram_total as f64) * 100.0).clamp(0.0, 100.0) as u16
    } else {
        0
    };
    let ram_color = if ram_pct >= 90 {
        theme::ERROR
    } else if ram_pct >= 75 {
        theme::WARNING
    } else {
        theme::SUCCESS
    };
    let ram_gauge = Gauge::default()
        .gauge_style(Style::default().fg(ram_color).bg(theme::BG_HIGHLIGHT))
        .percent(ram_pct)
        .label(format!(
            "RAM {} free / {} ({}T CPU)",
            fmt_mb(ram_avail),
            fmt_mb(ram_total),
            state.hardware.cpus
        ));
    f.render_widget(ram_gauge, gauge_cols[1]);

    if let Some(gpu) = state
        .hardware
        .gpus
        .iter()
        .find(|g| Some(g.index) == state.gpu_device)
        .or_else(|| state.hardware.gpus.first())
    {
        let v_total = gpu.vram_total_mb;
        let v_used = gpu.vram_used_mb;
        let v_free = v_total.saturating_sub(v_used);
        let v_pct = if v_total > 0 {
            ((v_used as f64 / v_total as f64) * 100.0).clamp(0.0, 100.0) as u16
        } else {
            0
        };
        let v_color = if v_pct >= 90 {
            theme::ERROR
        } else if v_pct >= 75 {
            theme::WARNING
        } else {
            theme::REASONING
        };
        let vram_gauge = Gauge::default()
            .gauge_style(Style::default().fg(v_color).bg(theme::BG_HIGHLIGHT))
            .percent(v_pct)
            .label(format!(
                "VRAM {} free / {} ({})",
                fmt_mb(v_free),
                fmt_mb(v_total),
                gpu.name
            ));
        f.render_widget(vram_gauge, gauge_cols[3]);
    } else {
        let cpu_only = Paragraph::new(Line::styled(
            "GPU: None detected (CPU-only)",
            Style::default().fg(theme::TEXT_MUTED),
        ));
        f.render_widget(cpu_only, gauge_cols[3]);
    }

    // Row 2: Est. load verdict
    f.render_widget(Paragraph::new(render_load_estimate_line(state)), rows[2]);

    // Row 3: Pinned Quick Actions bar
    let mut actions = vec![Span::raw(" ")];
    let start_label = if state.phase == ServerPhase::Running {
        "Stop Server"
    } else if state.phase == ServerPhase::LoadingModel {
        "Cancel"
    } else {
        "Start Server"
    };
    actions.extend(theme::keycap("s", start_label));
    actions.extend(theme::keycap("l", "Load Model"));
    actions.extend(theme::keycap("u", "Unload"));
    actions.extend(theme::keycap("k", "API Keys"));
    actions.extend(theme::keycap("[/]", "Sub-tab"));
    f.render_widget(Paragraph::new(Line::from(actions)), rows[3]);
}

fn render_server_subtab_bar(f: &mut Frame, area: Rect, active: ServerSubTab) {
    let mut spans = vec![Span::raw(" ")];
    for tab in ServerSubTab::ALL {
        if tab == active {
            spans.push(Span::styled(
                format!(" {} ", tab.label()),
                Style::default()
                    .fg(theme::BG_HEADER)
                    .bg(theme::PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(
                format!(" {} ", tab.label()),
                Style::default().fg(theme::TEXT_DIM),
            ));
        }
        spans.push(Span::styled(
            " │ ",
            Style::default().fg(theme::BG_HIGHLIGHT),
        ));
    }
    let p = Paragraph::new(Line::from(spans)).style(Style::default().bg(theme::BG_SURFACE));
    f.render_widget(p, area);
}

/// Map a mouse click column on the Server Panel sub-tab bar to a `ServerSubTab`.
pub fn server_subtab_at_col(col: u16) -> Option<ServerSubTab> {
    let mut x = 1u16;
    for tab in ServerSubTab::ALL {
        let w = (tab.label().len() + 2) as u16;
        if col >= x && col < x + w {
            return Some(tab);
        }
        x += w + 3; // " │ "
    }
    None
}

fn render_load_estimate_line(state: &ServerPanelState) -> Line<'static> {
    let label = Span::styled(" Est. load: ", Style::default().fg(theme::WARNING));

    let Some(est) = state.estimate_selected_model_load() else {
        return Line::from(vec![
            label,
            Span::styled("no model selected", Style::default().fg(theme::TEXT_MUTED)),
        ]);
    };

    let (verdict, verdict_color) = if !est.fits {
        ("✗ OOM", theme::ERROR)
    } else if est.tight {
        ("⚠ TIGHT", theme::WARNING)
    } else {
        ("✓ FITS", theme::SUCCESS)
    };

    let mut spans = vec![label];

    if est.full_gpu_offload {
        spans.push(Span::styled(
            format!(
                "~{} VRAM + ~{} RAM",
                fmt_mb(est.vram_mb.unwrap_or(0)),
                fmt_mb(est.ram_mb),
            ),
            Style::default().fg(theme::TEXT),
        ));
    } else if est.partial_gpu_offload {
        spans.push(Span::styled(
            format!(
                "≤{} RAM + ~{} VRAM ({}L GPU)",
                fmt_mb(est.ram_mb),
                fmt_mb(est.vram_mb.unwrap_or(0)),
                est.gpu_layers,
            ),
            Style::default().fg(theme::TEXT),
        ));
    } else {
        spans.push(Span::styled(
            format!(
                "~{} RAM (model {} + ctx {})",
                fmt_mb(est.ram_mb),
                fmt_mb(est.model_size_mb),
                fmt_mb(est.ram_mb.saturating_sub(est.model_size_mb)),
            ),
            Style::default().fg(theme::TEXT),
        ));
    }

    spans.push(Span::raw("  "));
    spans.push(theme::pill(verdict, theme::BG_HEADER, verdict_color));

    Line::from(spans)
}

fn render_config_fields(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let title = format!(" {} Configuration ", state.active_subtab.label());
    let block = theme::rounded_block(title, !state.editing);

    let visible_indices = state.visible_field_indices();
    let mut lines: Vec<Line> = Vec::new();
    let mut current_section = "";
    let mut selected_line: u16 = 0;

    for &i in &visible_indices {
        let field = &state.fields[i];
        let section = field.section();
        if section != current_section {
            current_section = section;
            if !lines.is_empty() {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {} ", section),
                    Style::default()
                        .fg(theme::WARNING)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "────────────────────────────────────",
                    Style::default().fg(theme::BG_HIGHLIGHT),
                ),
            ]));
        }

        let is_selected = i == state.selected;
        if is_selected {
            selected_line = lines.len() as u16;
        }
        let prefix = if is_selected { " ▶ " } else { "   " };
        let row_bg = if is_selected {
            theme::BG_HIGHLIGHT
        } else {
            Color::Reset
        };

        if *field == ConfigField::ModelSelection {
            let value = if state.models.is_empty() {
                "No models found — press F3 to download".to_string()
            } else {
                state
                    .models
                    .get(state.model_selected)
                    .map(|m| {
                        let q = m
                            .quantization
                            .as_ref()
                            .map(|q| format!(" [{}]", q))
                            .unwrap_or_default();
                        format!(
                            "{}{} ({})",
                            m.name,
                            q,
                            format_bytes_binary(m.file_size_bytes)
                        )
                    })
                    .unwrap_or_default()
            };

            lines.push(Line::from(vec![
                Span::styled(prefix, Style::default().fg(theme::PRIMARY).bg(row_bg)),
                Span::styled(
                    format!("{:<22}: ", field.label()),
                    Style::default()
                        .fg(if is_selected {
                            theme::TEXT
                        } else {
                            theme::TEXT_DIM
                        })
                        .bg(row_bg)
                        .add_modifier(if is_selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(
                    value,
                    Style::default()
                        .fg(theme::PRIMARY)
                        .bg(row_bg)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            if is_selected && !state.editing {
                lines.push(Line::styled(
                    "      hint: ←/→ to cycle local models",
                    Style::default().fg(theme::TEXT_MUTED),
                ));
            }
            continue;
        }

        if field.is_action() {
            let (action_color, action_text): (Color, String) = match field {
                ConfigField::StartServer => match state.phase {
                    ServerPhase::Running => (theme::TEXT_MUTED, "● Server is running".to_string()),
                    ServerPhase::LoadingModel => (theme::PRIMARY, "⟳ Loading model…".to_string()),
                    _ => (theme::SUCCESS, "▶ Start Server".to_string()),
                },
                ConfigField::StopServer => match state.phase {
                    ServerPhase::Running => (theme::ERROR, "■ Stop Server".to_string()),
                    ServerPhase::LoadingModel => (theme::WARNING, "■ Cancel Loading".to_string()),
                    _ => (theme::TEXT_MUTED, "■ Server not running".to_string()),
                },
                ConfigField::LoadAdditionalModel => match state.phase {
                    ServerPhase::LoadingModel => (theme::PRIMARY, "⟳ Loading model…".to_string()),
                    ServerPhase::Running => (theme::SUCCESS, "▶ Load Additional Model".to_string()),
                    _ => (theme::TEXT_MUTED, "○ Start server first".to_string()),
                },
                ConfigField::UnloadModel => {
                    if state.loaded_models.is_empty() {
                        (theme::TEXT_MUTED, "○ No models loaded".to_string())
                    } else {
                        let m = &state.loaded_models[state
                            .unload_model_selected
                            .min(state.loaded_models.len() - 1)];
                        (
                            theme::WARNING,
                            format!(
                                "■ Unload: {}{}",
                                m.name,
                                if m.is_default { " [default]" } else { "" }
                            ),
                        )
                    }
                }
                ConfigField::SetDefaultModel => {
                    if state.loaded_models.is_empty() {
                        (theme::TEXT_MUTED, "○ No models loaded".to_string())
                    } else {
                        let m = &state.loaded_models[state
                            .default_model_selected
                            .min(state.loaded_models.len() - 1)];
                        (theme::PRIMARY, format!("★ Default: {}", m.name))
                    }
                }
                ConfigField::ManageApiKeys => {
                    if state.api_keys.is_empty() {
                        (theme::PRIMARY, "◆ API Keys (Enter to manage)".to_string())
                    } else {
                        (
                            theme::PRIMARY,
                            format!("◆ {} API key(s) (Enter to manage)", state.api_keys.len()),
                        )
                    }
                }
                _ => (theme::TEXT_DIM, String::new()),
            };

            lines.push(Line::from(vec![
                Span::styled(prefix, Style::default().fg(theme::PRIMARY).bg(row_bg)),
                Span::styled(
                    action_text,
                    Style::default()
                        .fg(action_color)
                        .bg(row_bg)
                        .add_modifier(if is_selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
            ]));
            continue;
        }

        let label_span = Span::styled(
            format!("{:<22}: ", field.label()),
            Style::default()
                .fg(if is_selected {
                    theme::TEXT
                } else {
                    theme::TEXT_DIM
                })
                .bg(row_bg)
                .add_modifier(if is_selected {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        );

        let value_span = if state.editing && is_selected {
            Span::styled(
                format!("{}▏", state.edit_buffer),
                Style::default()
                    .fg(theme::TEXT)
                    .bg(theme::BG_SURFACE)
                    .add_modifier(Modifier::BOLD),
            )
        } else if field.is_toggle() {
            let v = state.field_value(field);
            if v == "ON" {
                theme::pill("ON", theme::BG_HEADER, theme::SUCCESS)
            } else {
                theme::pill("OFF", theme::TEXT_DIM, theme::BG_HIGHLIGHT)
            }
        } else {
            Span::styled(
                state.field_value(field),
                Style::default().fg(theme::PRIMARY).bg(row_bg),
            )
        };

        lines.push(Line::from(vec![
            Span::styled(prefix, Style::default().fg(theme::PRIMARY).bg(row_bg)),
            label_span,
            value_span,
        ]));

        if is_selected && !state.editing {
            lines.push(Line::styled(
                format!("      hint: {}", state.field_hint(field)),
                Style::default().fg(theme::TEXT_MUTED),
            ));
        }
    }

    let total_lines = lines.len();
    let visible_height = area.height.saturating_sub(2);
    let selected_end = selected_line + 1;
    let scroll = if selected_end >= visible_height {
        selected_end - visible_height + 1
    } else {
        0
    };

    let p = Paragraph::new(lines)
        .block(block)
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
    if total_lines > visible_height as usize {
        theme::render_scrollbar(
            f,
            area,
            total_lines.saturating_sub(visible_height as usize) + 1,
            scroll as usize,
        );
    }
}

fn render_loaded_models_panel(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let block = theme::rounded_block(" Loaded Models ", false);
    let mut lines: Vec<Line> = Vec::new();
    for m in &state.loaded_models {
        let mut spans = vec![
            Span::styled(
                format!(" {:>2} ", m.id),
                Style::default().fg(theme::TEXT_MUTED),
            ),
            Span::styled(
                m.name.clone(),
                Style::default()
                    .fg(if m.is_default {
                        theme::SUCCESS
                    } else {
                        theme::TEXT
                    })
                    .add_modifier(if m.is_default {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
            Span::styled(
                format!("  [{}]", m.backend),
                Style::default().fg(theme::SECONDARY),
            ),
        ];
        if m.is_default {
            spans.push(Span::styled(
                "  ★ default",
                Style::default().fg(theme::WARNING),
            ));
        }
        lines.push(Line::from(spans));
    }
    let p = Paragraph::new(lines).block(block);
    f.render_widget(p, area);
}

fn render_server_status_bar(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let (status, color) = if state.editing {
        (
            " Enter: Save │ Esc: Cancel │ Type to edit ".to_string(),
            theme::PRIMARY,
        )
    } else if let Some(ref msg) = state.status_message {
        let c = if state.status_is_error {
            theme::ERROR
        } else {
            theme::SUCCESS
        };
        (msg.lines().next().unwrap_or(msg).to_string(), c)
    } else if state.phase == ServerPhase::Running {
        (
            format!(
                " ● Running on {} │ [s] Stop · [l] Load · [u] Unload · [k] Keys · [/] or Ctrl+←/→ Sub-tab ",
                state.server_url.as_deref().unwrap_or("?")
            ),
            theme::SUCCESS,
        )
    } else {
        (
            " ↑/↓ Nav │ ←/→ Cycle/Toggle │ Enter Edit │ [/] or Ctrl+←/→ Sub-tab │ [s] Start │ [k] Keys "
                .to_string(),
            theme::TEXT_DIM,
        )
    };

    let status_bar = Paragraph::new(status)
        .style(Style::default().fg(color).bg(theme::BG_HEADER))
        .alignment(Alignment::Center);
    f.render_widget(status_bar, area);
}

// ============================================================================
// SETTINGS TAB (F5) — Full Scrolling + Scrollbar + Pill Toggles + No Box::leak
// ============================================================================

pub fn render_settings(f: &mut Frame, area: Rect, state: &SettingsState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);

    let block = theme::rounded_block(
        " ◆ Settings — ←/→ or Enter/Space to toggle/cycle · Enter to edit ",
        !state.editing,
    );

    let mut lines: Vec<Line> = Vec::new();
    let mut selected_line: u16 = 0;

    if let Some(ref hw) = state.hardware {
        lines.push(Line::from(vec![
            Span::styled(
                " Hardware ",
                Style::default()
                    .fg(theme::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "─────────────────────────────────────────",
                Style::default().fg(theme::BG_HIGHLIGHT),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("   CPU: ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(
                format!("{} threads", hw.cpus),
                Style::default().fg(theme::TEXT),
            ),
            Span::styled("   RAM: ", Style::default().fg(theme::TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} free / {}",
                    fmt_mb(hw.memory_available_mb),
                    fmt_mb(hw.memory_total_mb)
                ),
                Style::default().fg(theme::TEXT),
            ),
        ]));
        if hw.gpus.is_empty() {
            lines.push(Line::styled(
                "   GPU: None detected (CPU-only mode)",
                Style::default().fg(theme::WARNING),
            ));
        } else {
            for gpu in &hw.gpus {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("   GPU {}: ", gpu.index),
                        Style::default().fg(theme::TEXT_MUTED),
                    ),
                    Span::styled(
                        format!(
                            "{} ({} VRAM, {} used)",
                            gpu.name,
                            fmt_mb(gpu.vram_total_mb),
                            fmt_mb(gpu.vram_used_mb),
                        ),
                        Style::default().fg(theme::REASONING),
                    ),
                ]));
            }
        }
        lines.push(Line::from(""));
    }

    let mut current_section = "";
    for (i, field) in state.fields.iter().enumerate() {
        let section = field.section();
        if section != current_section {
            current_section = section;
            if !lines.is_empty() {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {} ", section),
                    Style::default()
                        .fg(theme::WARNING)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "───────────────────────────────────────",
                    Style::default().fg(theme::BG_HIGHLIGHT),
                ),
            ]));
        }

        let is_selected = i == state.selected;
        if is_selected {
            selected_line = lines.len() as u16;
        }
        let prefix = if is_selected { " ▶ " } else { "   " };
        let row_bg = if is_selected {
            theme::BG_HIGHLIGHT
        } else {
            Color::Reset
        };

        let label_span = Span::styled(
            format!("{:<22}: ", field.label()),
            Style::default()
                .fg(if is_selected {
                    theme::TEXT
                } else {
                    theme::TEXT_DIM
                })
                .bg(row_bg)
                .add_modifier(if is_selected {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        );

        let value_span = if state.editing && is_selected {
            Span::styled(
                format!("{}▏", state.edit_buffer),
                Style::default()
                    .fg(theme::TEXT)
                    .bg(theme::BG_SURFACE)
                    .add_modifier(Modifier::BOLD),
            )
        } else if field.is_toggle() {
            let v = state.field_value(field);
            if v == "ON" {
                theme::pill("ON", theme::BG_HEADER, theme::SUCCESS)
            } else {
                theme::pill("OFF", theme::TEXT_DIM, theme::BG_HIGHLIGHT)
            }
        } else {
            Span::styled(
                state.field_value(field),
                Style::default().fg(theme::PRIMARY).bg(row_bg),
            )
        };

        lines.push(Line::from(vec![
            Span::styled(prefix, Style::default().fg(theme::PRIMARY).bg(row_bg)),
            label_span,
            value_span,
        ]));

        if is_selected && !state.editing {
            lines.push(Line::styled(
                format!("      hint: {}", state.field_hint(field)),
                Style::default().fg(theme::TEXT_MUTED),
            ));
        }
    }

    let total_lines = lines.len();
    let visible_height = chunks[0].height.saturating_sub(2);
    let selected_end = selected_line + 1;
    let scroll = if selected_end >= visible_height {
        selected_end - visible_height + 1
    } else {
        0
    };

    let p = Paragraph::new(lines)
        .block(block)
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(p, chunks[0]);
    if total_lines > visible_height as usize {
        theme::render_scrollbar(
            f,
            chunks[0],
            total_lines.saturating_sub(visible_height as usize) + 1,
            scroll as usize,
        );
    }

    // Status bar — NO Box::leak! Use &str directly.
    let (status_text, status_color) = if state.editing {
        (" Enter: Save │ Esc: Cancel │ Type to edit ", theme::PRIMARY)
    } else if let Some(ref msg) = state.status_message {
        let color = if state.status_is_error {
            theme::ERROR
        } else {
            theme::SUCCESS
        };
        (msg.as_str(), color)
    } else {
        (
            " ↑/↓ Nav · PgUp/PgDn ±5 │ ←/→ Toggle/Cycle │ Enter Edit │ Esc Chat │ ? Help ",
            theme::TEXT_DIM,
        )
    };

    let status_bar = Paragraph::new(status_text)
        .style(Style::default().fg(status_color).bg(theme::BG_HEADER))
        .alignment(Alignment::Center);
    f.render_widget(status_bar, chunks[1]);
}

// ============================================================================
// LOGS TAB (F6) — Level Filter Bar, Live Search, Wrap Toggle, Scrollbar
// ============================================================================

pub fn render_logs(f: &mut Frame, area: Rect, state: &LogsState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Filter & search bar
            Constraint::Min(3),    // Log stream
            Constraint::Length(1), // Status bar
        ])
        .split(area);

    // Top filter bar
    let mut filter_spans: Vec<Span> = vec![Span::raw(" ")];
    let levels = [
        ("a", LogLevelFilter::All, theme::PRIMARY),
        ("e", LogLevelFilter::Error, theme::ERROR),
        ("w", LogLevelFilter::Warn, theme::WARNING),
        ("i", LogLevelFilter::Info, theme::SUCCESS),
        ("d", LogLevelFilter::Debug, theme::SECONDARY),
    ];
    for (key, lvl, color) in levels {
        if state.level_filter == lvl {
            filter_spans.push(theme::pill(
                format!("[{}] {}", key, lvl.label()),
                theme::BG_HEADER,
                color,
            ));
        } else {
            filter_spans.push(Span::styled(
                format!(" [{}] {} ", key, lvl.label()),
                Style::default().fg(theme::TEXT_DIM),
            ));
        }
        filter_spans.push(Span::raw(" "));
    }

    // Search indicator
    if state.searching {
        filter_spans.push(Span::styled(" │ /", Style::default().fg(theme::PRIMARY)));
        filter_spans.push(Span::styled(
            format!("{}▏", state.search_query),
            Style::default()
                .fg(theme::TEXT)
                .bg(theme::BG_HIGHLIGHT)
                .add_modifier(Modifier::BOLD),
        ));
    } else if !state.search_query.is_empty() {
        filter_spans.push(Span::styled(
            format!(" │ /{}", state.search_query),
            Style::default().fg(theme::PRIMARY),
        ));
    }

    let entries = state.filtered_entries();
    let count_str = format!(
        " {} / {} {} ",
        entries.len(),
        state.max_entries(),
        if state.wrap_lines { "[WRAP]" } else { "" }
    );
    let left_len: usize = filter_spans.iter().map(|s| s.content.chars().count()).sum();
    let right_len = count_str.chars().count();
    let pad = (chunks[0].width as usize).saturating_sub(left_len + right_len);
    filter_spans.push(Span::raw(" ".repeat(pad)));
    filter_spans.push(Span::styled(
        count_str,
        Style::default().fg(theme::TEXT_MUTED),
    ));
    f.render_widget(
        Paragraph::new(Line::from(filter_spans)).style(Style::default().bg(theme::BG_SURFACE)),
        chunks[0],
    );

    // Log stream pane
    let title = format!(
        " ◆ Live Logs — {} ({}) ",
        state.level_filter.label(),
        if state.auto_scroll { "LIVE" } else { "PAUSED" }
    );
    let block = theme::rounded_block(title, !state.searching);
    let inner_width = chunks[1].width.saturating_sub(3) as usize;

    let mut lines: Vec<Line> = Vec::with_capacity(entries.len());
    for entry in &entries {
        let level_color = match entry.level.as_str() {
            "ERROR" => theme::ERROR,
            "WARN" => theme::WARNING,
            "INFO" => theme::SUCCESS,
            "DEBUG" => theme::SECONDARY,
            "TRACE" => theme::TEXT_MUTED,
            "LOG" => theme::PRIMARY,
            _ => theme::TEXT,
        };

        let prefix_len = 20usize;
        let max_target = 22usize;
        let target_display: String = if entry.target.chars().count() > max_target {
            entry.target.chars().take(max_target).collect()
        } else {
            entry.target.clone()
        };
        let target_span_len = target_display.chars().count() + 1;

        let message_display = if state.wrap_lines {
            entry.message.clone()
        } else {
            let max_msg = inner_width.saturating_sub(prefix_len + target_span_len);
            if entry.message.chars().count() > max_msg && max_msg > 3 {
                let truncated: String = entry.message.chars().take(max_msg - 1).collect();
                format!("{}…", truncated)
            } else {
                entry.message.clone()
            }
        };

        lines.push(Line::from(vec![
            Span::styled(
                format!(" {} ", entry.timestamp),
                Style::default().fg(theme::TEXT_MUTED),
            ),
            Span::styled(
                format!("{:<5} ", entry.level),
                Style::default()
                    .fg(level_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{} ", target_display),
                Style::default().fg(theme::TEXT_MUTED),
            ),
            Span::styled(message_display, Style::default().fg(theme::TEXT)),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::styled(
            "  No matching logs yet. Logs appear here in real-time.",
            Style::default().fg(theme::TEXT_MUTED),
        ));
    }

    let total_lines = lines.len();
    let visible_height = chunks[1].height.saturating_sub(2) as usize;
    let max_scroll = total_lines.saturating_sub(visible_height);

    let scroll = if state.auto_scroll {
        max_scroll
    } else {
        (state.scroll_top as usize).min(max_scroll)
    };

    let mut p = Paragraph::new(lines)
        .block(block)
        .scroll((scroll.min(u16::MAX as usize) as u16, 0));
    if state.wrap_lines {
        p = p.wrap(Wrap { trim: false });
    }
    f.render_widget(p, chunks[1]);
    if total_lines > visible_height {
        theme::render_scrollbar(f, chunks[1], max_scroll + 1, scroll);
    }

    // Bottom status bar
    let mut hints = Vec::new();
    hints.extend(theme::keycap("a/e/w/i/d", "Level"));
    hints.extend(theme::keycap("/", "Search"));
    hints.extend(theme::keycap("z", "Wrap"));
    hints.extend(theme::keycap("End", "Live"));
    hints.extend(theme::keycap("c", "Clear"));
    let status_bar = Paragraph::new(Line::from(hints))
        .style(Style::default().bg(theme::BG_HEADER))
        .alignment(Alignment::Center);
    f.render_widget(status_bar, chunks[2]);
}
