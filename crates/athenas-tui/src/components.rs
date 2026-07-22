use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::chat::ChatState;
use crate::model_browser::{BrowserPhase, ModelBrowserState};
use crate::server_panel::{ConfigField, ServerPanelState, ServerPhase};
use crate::settings::SettingsState;

pub fn render_chat_area(
    f: &mut Frame,
    area: Rect,
    state: &mut ChatState,
    is_loading_model: bool,
    loading_spinner: usize,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    render_messages(f, chunks[0], state, is_loading_model, loading_spinner);
    render_input(f, chunks[1], state);
    render_status_bar(f, chunks[2], state);
}

fn render_messages(
    f: &mut Frame,
    area: Rect,
    state: &mut ChatState,
    is_loading_model: bool,
    loading_spinner: usize,
) {
    let mut lines: Vec<Line> = Vec::new();

    for msg in &state.messages {
        let (role_color, role_str) = match msg.role.as_str() {
            "user" => (Color::Green, "You"),
            "assistant" => (Color::Cyan, "AI"),
            "system" => (Color::Yellow, "System"),
            _ => (Color::Gray, msg.role.as_str()),
        };

        lines.push(Line::styled(
            format!(" {} ", role_str),
            Style::default().fg(role_color).add_modifier(Modifier::BOLD),
        ));

        // Render collapsible reasoning section for assistant messages
        if msg.role == "assistant" && !msg.reasoning.is_empty() {
            if msg.reasoning_expanded {
                lines.push(Line::styled(
                    "  [Thinking] ▼ (Tab to collapse)",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::DIM),
                ));
                for line in msg.reasoning.lines() {
                    lines.push(Line::styled(
                        format!("    {}", line),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::DIM),
                    ));
                }
                lines.push(Line::styled(
                    "  [/Thinking]",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::DIM),
                ));
            } else {
                let preview_len = 60;
                let preview: String = msg.reasoning.chars().take(preview_len).collect();
                let suffix = if msg.reasoning.chars().count() > preview_len {
                    "..."
                } else {
                    ""
                };
                lines.push(Line::styled(
                    format!("  [Thinking] ▶ {}{} (Tab to expand)", preview, suffix),
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::DIM),
                ));
            }
        }

        for line in msg.content.lines() {
            lines.push(Line::from(format!("  {}", line)));
        }
        lines.push(Line::from(""));
    }

    if state.is_generating {
        let elapsed = state
            .generation_start
            .map(|s| s.elapsed().as_secs())
            .unwrap_or(0);

        lines.push(Line::styled(
            " AI",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        if state.streaming_text.is_empty() && state.streaming_reasoning.is_empty() {
            // Show animated spinner + elapsed time while waiting for first token
            let spinner_char = match elapsed % 4 {
                0 => "|",
                1 => "/",
                2 => "-",
                _ => "\\",
            };
            let wait_msg = if elapsed > 60 {
                format!(
                    "  {} Still waiting... {}s (model may be thinking or stuck)",
                    spinner_char, elapsed
                )
            } else if elapsed > 30 {
                format!("  {} Waiting for response... {}s", spinner_char, elapsed)
            } else if elapsed > 5 {
                format!("  {} Processing... {}s", spinner_char, elapsed)
            } else {
                format!("  {} Generating...", spinner_char)
            };
            let wait_color = if elapsed > 60 {
                Color::Yellow
            } else {
                Color::Cyan
            };
            lines.push(Line::styled(
                wait_msg,
                Style::default()
                    .fg(wait_color)
                    .add_modifier(Modifier::ITALIC),
            ));
        } else {
            // Show live reasoning if present
            if !state.streaming_reasoning.is_empty() {
                lines.push(Line::styled(
                    "  [Thinking] ▼ (live...)",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::DIM),
                ));
                for line in state.streaming_reasoning.lines() {
                    lines.push(Line::styled(
                        format!("    {}", line),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::DIM),
                    ));
                }
                lines.push(Line::styled(
                    "  [/Thinking]",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::DIM),
                ));
            }
            for line in state.streaming_text.lines() {
                lines.push(Line::from(format!("  {}", line)));
            }
            // Show live elapsed + tok/s during streaming
            if elapsed > 2 {
                let info = if let Some(tps) = state.tokens_per_second {
                    format!("  ~{:.1} tok/s · {}s", tps, elapsed)
                } else {
                    format!("  {}s", elapsed)
                };
                lines.push(Line::styled(
                    info,
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::ITALIC),
                ));
            }
        }
        lines.push(Line::from(""));
    }

    if is_loading_model {
        let spinner = match loading_spinner {
            0 => "|",
            1 => "/",
            2 => "-",
            _ => "\\",
        };
        lines.push(Line::styled(
            format!(" {} Loading model... Please wait", spinner),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }

    // Calculate visible area height (inside borders)
    let inner_height = area.height.saturating_sub(2) as usize;
    let total_lines = lines.len();
    let max_scroll = total_lines.saturating_sub(inner_height);

    // Save max_scroll so key handler can detect bottom
    state.max_scroll = max_scroll;

    // If not auto-scrolling and user scrolled past bottom, re-enable auto-scroll
    if !state.auto_scroll && state.scroll >= max_scroll {
        state.auto_scroll = true;
    }

    // Auto-scroll to bottom when enabled; clamp manual scroll to max
    let scroll = if state.auto_scroll {
        max_scroll
    } else {
        state.scroll.min(max_scroll)
    };

    // Show scroll indicator in title when content overflows
    let title = if total_lines > inner_height {
        if state.auto_scroll {
            " Athenas Studio — Chat ".to_string()
        } else {
            let pct = if max_scroll > 0 {
                ((scroll as f32 / max_scroll as f32) * 100.0) as u32
            } else {
                0
            };
            format!(" Athenas Studio — Chat [{}%] ", pct)
        }
    } else {
        " Athenas Studio — Chat ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            title,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll as u16, 0));

    f.render_widget(paragraph, area);
}

fn render_input(f: &mut Frame, area: Rect, state: &ChatState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Input ")
        .border_style(Style::default().fg(Color::DarkGray));

    let input = Paragraph::new(state.input_text.as_str())
        .block(block)
        .style(Style::default().fg(Color::White));

    f.render_widget(input, area);

    let cursor_x = area.x + 1 + state.input_text.chars().count() as u16;
    let cursor_y = area.y + 1;
    f.set_cursor_position((cursor_x, cursor_y));
}

fn render_status_bar(f: &mut Frame, area: Rect, state: &ChatState) {
    let mut status_parts = Vec::new();

    if let Some(ref model) = state.current_model {
        status_parts.push(Span::styled(
            format!(" Model: {} ", model),
            Style::default().fg(Color::Cyan),
        ));
    } else {
        status_parts.push(Span::styled(
            " No model loaded ",
            Style::default().fg(Color::Red),
        ));
    }

    if let Some(ref backend) = state.current_backend {
        status_parts.push(Span::styled(
            format!(" Backend: {} ", backend),
            Style::default().fg(Color::Blue),
        ));
    }

    if let Some(tps) = state.tokens_per_second {
        status_parts.push(Span::styled(
            format!(" {:.1} tok/s ", tps),
            Style::default().fg(Color::Green),
        ));
    }

    status_parts.push(Span::raw(
        " | Enter: Send | Up/Down: Scroll | Tab: Thinking | Ctrl+C: Quit ",
    ));

    let line = Line::from(status_parts);
    let paragraph = Paragraph::new(line).style(Style::default().bg(Color::Black));
    f.render_widget(paragraph, area);
}

pub fn render_model_list(f: &mut Frame, area: Rect, state: &crate::model_list::ModelListState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Models (Enter: Load | Del: Delete) ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    if state.models.is_empty() {
        let p = Paragraph::new(
            "No models downloaded.\nUse 'athenas models pull <repo-id>' to download.",
        )
        .block(block)
        .style(Style::default().fg(Color::Gray));
        f.render_widget(p, area);
        return;
    }

    let items: Vec<ListItem> = state
        .models
        .iter()
        .map(|m| {
            let mut spans = vec![Span::styled(
                m.name.clone(),
                Style::default().fg(Color::White),
            )];
            if let Some(ref q) = m.quantization {
                spans.push(Span::styled(
                    format!(" [{}]", q),
                    Style::default().fg(Color::Yellow),
                ));
            }
            spans.push(Span::styled(
                format!(" {}", m.format_size()),
                Style::default().fg(Color::Gray),
            ));
            ListItem::new(Line::from(spans))
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut state.list_state.clone());
}

pub fn render_tab_bar(f: &mut Frame, area: Rect, active: usize) {
    let tabs = ["Chat", "Models", "Browser", "Server", "Settings"];
    let spans: Vec<Span> = tabs
        .iter()
        .enumerate()
        .flat_map(|(i, label)| {
            let style = if i == active {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            vec![Span::styled(format!(" {} ", label), style), Span::raw(" ")]
        })
        .collect();
    let line = Line::from(spans);
    let p = Paragraph::new(line).style(Style::default().bg(Color::Black));
    f.render_widget(p, area);
}

pub fn render_settings(f: &mut Frame, area: Rect, state: &SettingsState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Settings — Enter to edit, Esc to cancel, Enter to save ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    let mut lines: Vec<Line> = Vec::new();
    let mut current_section = "";

    for (i, field) in state.fields.iter().enumerate() {
        let section = field.section();
        if section != current_section {
            current_section = section;
            lines.push(Line::from(""));
            lines.push(Line::styled(
                format!(" {} ", section),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::styled(
                " ───────────────────────────────────────",
                Style::default().fg(Color::DarkGray),
            ));
        }

        let is_selected = i == state.selected;
        let prefix = if is_selected { " > " } else { "   " };
        let value = if state.editing && is_selected {
            format!("{}|", state.edit_buffer)
        } else {
            state.field_value(field)
        };

        let style = if is_selected {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };

        lines.push(Line::styled(
            format!("{}{: <16}: {}", prefix, field.label(), value),
            style,
        ));

        if is_selected && !state.editing {
            lines.push(Line::styled(
                format!("     hint: {}", state.field_hint(field)),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(p, chunks[0]);

    let status = if state.editing {
        " Enter: Save | Esc: Cancel | Type to edit "
    } else if let Some(ref msg) = state.status_message {
        Box::leak(msg.clone().into_boxed_str())
    } else {
        " Up/Down: Navigate | Enter: Edit | F2: Save all "
    };
    let status_bar = Paragraph::new(status)
        .style(Style::default().fg(Color::Cyan).bg(Color::Black))
        .alignment(Alignment::Center);
    f.render_widget(status_bar, chunks[1]);
}

pub fn render_model_browser(f: &mut Frame, area: Rect, state: &ModelBrowserState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Model Browser — Search & Download from HuggingFace ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    let mut lines: Vec<Line> = Vec::new();

    match &state.phase {
        BrowserPhase::Search => {
            lines.push(Line::styled(
                " Search HuggingFace Models",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(""));
            lines.push(Line::styled(
                format!(" > {}|", state.search_input),
                Style::default().fg(Color::White),
            ));
            lines.push(Line::from(""));
            lines.push(Line::styled(
                format!(" GGUF only: {}", if state.gguf_only { "ON" } else { "OFF" }),
                Style::default().fg(if state.gguf_only {
                    Color::Green
                } else {
                    Color::Gray
                }),
            ));
            lines.push(Line::from(""));
            lines.push(Line::styled(
                " Enter to search | G: toggle GGUF filter | Ctrl+U: clear",
                Style::default().fg(Color::DarkGray),
            ));
        }
        BrowserPhase::Results => {
            lines.push(Line::styled(
                format!(
                    " Results for '{}' ({} found)",
                    state.search_input,
                    state.search_results.len()
                ),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(""));

            if state.search_results.is_empty() {
                lines.push(Line::styled(
                    " No models found. Try a different search.",
                    Style::default().fg(Color::Gray),
                ));
            } else {
                for (i, result) in state.search_results.iter().take(20).enumerate() {
                    let is_selected = i == state.results_selected;
                    let prefix = if is_selected { " > " } else { "   " };
                    let style = if is_selected {
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Gray)
                    };

                    let dl = if result.downloads >= 1000 {
                        format!("{:.1}k", result.downloads as f64 / 1000.0)
                    } else {
                        result.downloads.to_string()
                    };

                    lines.push(Line::styled(
                        format!(
                            "{}{} ({} dl, {} likes)",
                            prefix, result.id, dl, result.likes
                        ),
                        style,
                    ));
                }
                lines.push(Line::from(""));
                lines.push(Line::styled(
                    " Enter: Download | Esc or /: Edit search | R: Reset",
                    Style::default().fg(Color::DarkGray),
                ));
            }
        }
        BrowserPhase::SelectFile => {
            lines.push(Line::styled(
                " Select File to Download",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(""));

            for (i, (name, size)) in state.file_options.iter().enumerate() {
                let is_selected = i == state.file_selected;
                let prefix = if is_selected { " > " } else { "   " };
                let style = if is_selected {
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Gray)
                };

                let size_str = size
                    .map(|s| format!("{:.2} GB", s as f64 / 1e9))
                    .unwrap_or("?".to_string());

                lines.push(Line::styled(
                    format!("{}{} ({})", prefix, name, size_str),
                    style,
                ));
            }
            lines.push(Line::from(""));
            lines.push(Line::styled(
                " Enter: Download | Esc: Back to results",
                Style::default().fg(Color::DarkGray),
            ));
        }
        BrowserPhase::Downloading => {
            lines.push(Line::styled(
                " Downloading Model...",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::from(""));

            if let Some(ref name) = state.download_filename {
                lines.push(Line::styled(
                    format!(" File: {}", name),
                    Style::default().fg(Color::White),
                ));
            }

            if let Some((downloaded, total, speed_mbps)) = state.download_progress {
                let percent = if total > 0 {
                    (downloaded as f64 / total as f64 * 100.0) as u16
                } else {
                    0
                };
                let dl_str = format_bytes(downloaded);
                let total_str = if total > 0 {
                    format_bytes(total)
                } else {
                    "?".to_string()
                };

                lines.push(Line::from(""));
                lines.push(Line::styled(
                    format!(" {} / {} ({}%)", dl_str, total_str, percent),
                    Style::default().fg(Color::Cyan),
                ));
                lines.push(Line::styled(
                    format!(" Speed: {:.2} MB/s", speed_mbps),
                    Style::default().fg(Color::Green),
                ));

                let gauge = Gauge::default()
                    .block(Block::default().borders(Borders::ALL))
                    .gauge_style(Style::default().fg(Color::Cyan))
                    .percent(percent);
                let gauge_area = Rect::new(area.x + 2, area.y + 8, area.width - 4, 3);
                f.render_widget(gauge, gauge_area);
            } else {
                lines.push(Line::styled(
                    " Connecting...",
                    Style::default().fg(Color::Gray),
                ));
            }

            lines.push(Line::from(""));
            lines.push(Line::styled(
                " Please wait... (Esc to cancel)",
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    if let Some(ref msg) = state.status_message {
        lines.push(Line::from(""));
        let color = if state.status_is_error {
            Color::Red
        } else {
            Color::Green
        };
        let prefix = if state.status_is_error {
            "[!]"
        } else {
            "[✓]"
        };
        lines.push(Line::styled(
            format!(" {} {}", prefix, msg),
            Style::default().fg(color),
        ));
    }

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(p, chunks[0]);

    let help = match &state.phase {
        BrowserPhase::Search => " Enter: Search | G: Toggle GGUF | Esc: Back to Chat ",
        BrowserPhase::Results => " Up/Down: Navigate | Enter: Download | Esc: New search ",
        BrowserPhase::SelectFile => " Up/Down: Navigate | Enter: Download | Esc: Back ",
        BrowserPhase::Downloading => " Downloading... | Esc: Cancel ",
    };
    let status_bar = Paragraph::new(help)
        .style(Style::default().fg(Color::Cyan).bg(Color::Black))
        .alignment(Alignment::Center);
    f.render_widget(status_bar, chunks[1]);
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

pub fn render_server_panel(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7), // Hardware info banner
            Constraint::Min(3),    // Config fields
            Constraint::Length(3), // Status bar
        ])
        .split(area);

    render_hardware_banner(f, chunks[0], state);
    render_config_fields(f, chunks[1], state);
    render_server_status_bar(f, chunks[2], state);
}

fn render_hardware_banner(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Hardware ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    let mut lines: Vec<Line> = Vec::new();

    let cpu_threads = state.hardware.cpus;
    lines.push(Line::from(vec![
        Span::styled(" CPU: ", Style::default().fg(Color::Yellow)),
        Span::styled(
            format!("{} threads", cpu_threads),
            Style::default().fg(Color::White),
        ),
        Span::raw("  "),
        Span::styled("Mem: ", Style::default().fg(Color::Yellow)),
        Span::styled(
            format!("{} MB", state.hardware.memory_total_mb),
            Style::default().fg(Color::White),
        ),
    ]));

    let gpu_str = if state.hardware.gpus.is_empty() {
        "None (CPU-only)".to_string()
    } else {
        state
            .hardware
            .gpus
            .iter()
            .map(|g| format!("{} ({} MB)", g.name, g.vram_total_mb))
            .collect::<Vec<_>>()
            .join(", ")
    };
    lines.push(Line::from(vec![
        Span::styled(" GPU: ", Style::default().fg(Color::Yellow)),
        Span::styled(gpu_str, Style::default().fg(Color::White)),
    ]));

    let status_line = match &state.phase {
        ServerPhase::Configuring => Line::from(vec![
            Span::styled(" Status: ", Style::default().fg(Color::Yellow)),
            Span::styled("Configuring", Style::default().fg(Color::Gray)),
        ]),
        ServerPhase::LoadingModel => Line::from(vec![
            Span::styled(" Status: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                "Loading model...",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        ServerPhase::Running => {
            let url = state.server_url.as_deref().unwrap_or("?");
            let model = state.loaded_model_name.as_deref().unwrap_or("unknown");
            Line::from(vec![
                Span::styled(" Status: ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "RUNNING",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled("URL: ", Style::default().fg(Color::Yellow)),
                Span::styled(url, Style::default().fg(Color::Cyan)),
                Span::raw("  "),
                Span::styled("Model: ", Style::default().fg(Color::Yellow)),
                Span::styled(model, Style::default().fg(Color::White)),
            ])
        }
        ServerPhase::Error => Line::from(vec![
            Span::styled(" Status: ", Style::default().fg(Color::Yellow)),
            Span::styled("ERROR", Style::default().fg(Color::Red)),
        ]),
    };
    lines.push(status_line);

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn render_config_fields(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Server Configuration — Enter to edit/toggle, Up/Down to navigate ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    let mut lines: Vec<Line> = Vec::new();
    let mut current_section = "";

    for (i, field) in state.fields.iter().enumerate() {
        let section = field.section();
        if section != current_section {
            current_section = section;
            lines.push(Line::from(""));
            lines.push(Line::styled(
                format!(" {} ", section),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::styled(
                " ─────────────────────────────────────────────",
                Style::default().fg(Color::DarkGray),
            ));
        }

        let is_selected = i == state.selected;
        let prefix = if is_selected { " > " } else { "   " };

        // Special rendering for model selection
        if *field == ConfigField::ModelSelection {
            let value = if state.models.is_empty() {
                "No models found — use F3 to download".to_string()
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
                        format!("{}{} ({})", m.name, q, m.format_size())
                    })
                    .unwrap_or_default()
            };

            let style = if is_selected {
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };

            lines.push(Line::styled(
                format!("{}{: <16}: {}", prefix, field.label(), value),
                style,
            ));

            if is_selected && !state.editing {
                lines.push(Line::styled(
                    "     hint: Left/Right to cycle models",
                    Style::default().fg(Color::DarkGray),
                ));
            }
            continue;
        }

        // Special rendering for action buttons
        if field.is_action() {
            let is_start = *field == ConfigField::StartServer;
            let is_stop = *field == ConfigField::StopServer;

            let (action_color, action_text): (Color, String) = if is_start {
                if state.phase == ServerPhase::Running {
                    (Color::DarkGray, "● Server is running".to_string())
                } else if state.phase == ServerPhase::LoadingModel {
                    (Color::Cyan, "⟳ Loading model...".to_string())
                } else {
                    (Color::Green, "▶ Start Server".to_string())
                }
            } else if is_stop {
                if state.phase == ServerPhase::Running {
                    (Color::Red, "■ Stop Server".to_string())
                } else {
                    (Color::DarkGray, "■ Server not running".to_string())
                }
            } else if *field == ConfigField::LoadAdditionalModel {
                if state.phase == ServerPhase::Running {
                    (Color::Green, "▶ Load Additional Model".to_string())
                } else {
                    (Color::DarkGray, "○ Start server first".to_string())
                }
            } else if *field == ConfigField::UnloadModel {
                if state.loaded_models.is_empty() {
                    (Color::DarkGray, "○ No models loaded".to_string())
                } else {
                    let m = &state.loaded_models[state
                        .unload_model_selected
                        .min(state.loaded_models.len() - 1)];
                    (
                        Color::Yellow,
                        format!(
                            "■ Unload: {}{}",
                            m.name,
                            if m.is_default { " [default]" } else { "" }
                        ),
                    )
                }
            } else if *field == ConfigField::SetDefaultModel {
                if state.loaded_models.is_empty() {
                    (Color::DarkGray, "○ No models loaded".to_string())
                } else {
                    let m = &state.loaded_models[state
                        .default_model_selected
                        .min(state.loaded_models.len() - 1)];
                    (Color::Cyan, format!("★ Default: {}", m.name))
                }
            } else {
                (Color::Gray, String::new())
            };

            let style = if is_selected {
                Style::default()
                    .fg(action_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(action_color)
            };

            lines.push(Line::styled(format!("{}  {}", prefix, action_text), style));

            // Show hint for multi-model actions when selected
            if is_selected && !state.editing {
                let hint = match field {
                    ConfigField::LoadAdditionalModel => {
                        "Select model above with Left/Right, then press Enter here"
                    }
                    ConfigField::UnloadModel => "Left/Right to pick model, Enter to unload",
                    ConfigField::SetDefaultModel => {
                        "Left/Right to pick model, Enter to set default"
                    }
                    _ => "",
                };
                if !hint.is_empty() {
                    lines.push(Line::styled(
                        format!("     hint: {}", hint),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
            }
            continue;
        }

        let value = if state.editing && is_selected {
            format!("{}|", state.edit_buffer)
        } else {
            state.field_value(field)
        };

        let style = if is_selected {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };

        let value_style = if field.is_toggle() {
            let v = state.field_value(field);
            if v == "ON" {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::Red)
            }
        } else {
            Style::default().fg(Color::Cyan)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("{}{: <16}: ", prefix, field.label()), style),
            Span::styled(value, value_style),
        ]));

        if is_selected && !state.editing {
            lines.push(Line::styled(
                format!("     hint: {}", state.field_hint(field)),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    // Show loaded models and endpoints when running
    if state.phase == ServerPhase::Running {
        // Loaded models list
        if !state.loaded_models.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::styled(
                " LOADED MODELS",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
            lines.push(Line::styled(
                " ─────────────────────────────────────────────",
                Style::default().fg(Color::DarkGray),
            ));
            for m in &state.loaded_models {
                let default_marker = if m.is_default { " ★" } else { "" };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("   {}  ", m.id),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(
                        m.name.clone(),
                        Style::default().fg(if m.is_default {
                            Color::Green
                        } else {
                            Color::White
                        }),
                    ),
                    Span::styled(
                        format!("  [{}]", m.backend),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(default_marker, Style::default().fg(Color::Yellow)),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::styled(
            " ENDPOINTS",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::styled(
            "   POST /v1/chat/completions   GET /v1/models",
            Style::default().fg(Color::Gray),
        ));
        lines.push(Line::styled(
            "   POST /v1/completions        GET /v1/health",
            Style::default().fg(Color::Gray),
        ));
        lines.push(Line::styled(
            "   GET /v1/ready               GET /metrics",
            Style::default().fg(Color::Gray),
        ));
    }

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn render_server_status_bar(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let status = if state.editing {
        " Enter: Save | Esc: Cancel | Type to edit ".to_string()
    } else if let Some(ref msg) = state.status_message {
        msg.clone()
    } else if state.phase == ServerPhase::Running {
        format!(
            " Server running on {} | Enter on Stop to halt ",
            state.server_url.as_deref().unwrap_or("?")
        )
    } else {
        " Up/Down: Navigate | Enter: Edit/Toggle/Action | Left/Right: Cycle Model | F5: Server "
            .to_string()
    };

    let color = if state
        .status_message
        .as_ref()
        .is_some_and(|m| m.starts_with("Error") || m.starts_with("["))
    {
        Color::Red
    } else if state.phase == ServerPhase::Running {
        Color::Green
    } else {
        Color::Cyan
    };

    let status_bar = Paragraph::new(status)
        .style(Style::default().fg(color).bg(Color::Black))
        .alignment(Alignment::Center);
    f.render_widget(status_bar, area);
}

pub fn render_sidebar(
    f: &mut Frame,
    area: Rect,
    conversations: &[(String, String)],
    selected: usize,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Conversations ")
        .border_style(Style::default().fg(Color::DarkGray));

    let items: Vec<ListItem> = conversations
        .iter()
        .map(|(_id, title)| ListItem::new(Line::from(title.as_str())))
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    let mut state = ratatui::widgets::ListState::default();
    state.select(Some(selected));
    f.render_stateful_widget(list, area, &mut state);
}
