use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::chat::ChatState;
use crate::model_browser::{BrowserPhase, ModelBrowserState};
use crate::settings::SettingsState;

pub fn render_chat_area(f: &mut Frame, area: Rect, state: &ChatState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    render_messages(f, chunks[0], state);
    render_input(f, chunks[1], state);
    render_status_bar(f, chunks[2], state);
}

fn render_messages(f: &mut Frame, area: Rect, state: &ChatState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Athenas Studio — Chat ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

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

        for line in msg.content.lines() {
            lines.push(Line::from(format!("  {}", line)));
        }
        lines.push(Line::from(""));
    }

    if state.is_generating {
        lines.push(Line::styled(
            " AI is typing...",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::ITALIC),
        ));
    }

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((state.scroll as u16, 0));

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

    let cursor_x = area.x + 1 + state.input_text.len() as u16;
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

    status_parts.push(Span::raw(" | Enter: Send | Ctrl+C: Quit | Tab: Switch "));

    let line = Line::from(status_parts);
    let paragraph = Paragraph::new(line).style(Style::default().bg(Color::Black));
    f.render_widget(paragraph, area);
}

pub fn render_model_list(f: &mut Frame, area: Rect, state: &crate::model_list::ModelListState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Models ",
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
    let tabs = ["Chat", "Models", "Browser", "Settings"];
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
                " Press Enter to search | G: toggle GGUF filter",
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
                    " Enter: Download selected | Esc: New search",
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

            if let Some((downloaded, total)) = state.download_progress {
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
        lines.push(Line::styled(
            format!(" [!] {}", msg),
            Style::default().fg(Color::Red),
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
