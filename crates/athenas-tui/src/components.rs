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
    chat_input: &mut tui_textarea::TextArea<'static>,
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
    // Available width inside the block borders. We pre-wrap text to this
    // width so that lines.len() accurately reflects the number of rendered
    // rows. This fixes the scroll bug where wrapped lines weren't counted.
    //
    // -2 for left/right borders, -2 for the "  " indent prefix on content.
    let content_width = area.width.saturating_sub(4) as usize;
    let reasoning_width = area.width.saturating_sub(6) as usize; // "    " prefix

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
                    for wrapped in wrap_text(line, reasoning_width) {
                        lines.push(Line::styled(
                            format!("    {}", wrapped),
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::DIM),
                        ));
                    }
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
            for wrapped in wrap_text(line, content_width) {
                lines.push(Line::from(format!("  {}", wrapped)));
            }
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
                    for wrapped in wrap_text(line, reasoning_width) {
                        lines.push(Line::styled(
                            format!("    {}", wrapped),
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::DIM),
                        ));
                    }
                }
                lines.push(Line::styled(
                    "  [/Thinking]",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::DIM),
                ));
            }
            for line in state.streaming_text.lines() {
                for wrapped in wrap_text(line, content_width) {
                    lines.push(Line::from(format!("  {}", wrapped)));
                }
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

    // No .wrap() — we pre-wrapped the text ourselves so lines.len()
    // matches the actual rendered row count, making scroll accurate.
    let paragraph = Paragraph::new(lines)
        .block(block)
        .scroll((scroll as u16, 0));

    f.render_widget(paragraph, area);
}

/// Wrap a single line of text to fit within `width` display columns.
/// Respects word boundaries — breaks at spaces when possible, and breaks
/// mid-word when a single word is longer than `width`.
///
/// Returns one or more strings, each fitting within `width` columns.
/// If the input is empty, returns a single empty string.
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
            // First word on this line
            if word_len <= width {
                current = word.to_string();
                current_len = word_len;
            } else {
                // Word is longer than width — break it character by character
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
            // Word fits on current line
            current.push(' ');
            current.push_str(word);
            current_len += 1 + word_len;
        } else {
            // Word doesn't fit — flush current line and start new one
            result.push(current.clone());
            current.clear();
            current_len = 0;

            if word_len <= width {
                current = word.to_string();
                current_len = word_len;
            } else {
                // Word is longer than width — break it
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
            format!(" {} ", backend),
            Style::default().fg(Color::Blue),
        ));
    }

    // GPU info: show GPU name + runtime + layers, or CPU
    if !state.gpu_info.is_empty() {
        let layers_str = if state.gpu_layers < 0 {
            "all".to_string()
        } else if state.gpu_layers == 0 {
            "CPU".to_string()
        } else {
            state.gpu_layers.to_string()
        };
        status_parts.push(Span::styled(
            format!(
                " GPU: {} [{}] {} layers ",
                state.gpu_info, state.gpu_runtime, layers_str
            ),
            Style::default().fg(Color::Magenta),
        ));
    } else if state.gpu_layers == 0 {
        status_parts.push(Span::styled(
            " CPU mode ",
            Style::default().fg(Color::Yellow),
        ));
    }

    if let Some(tps) = state.tokens_per_second {
        status_parts.push(Span::styled(
            format!(" {:.1} tok/s ", tps),
            Style::default().fg(Color::Green),
        ));
    }

    if !state.system_prompt.is_empty() {
        status_parts.push(Span::styled(
            " [system prompt] ",
            Style::default().fg(Color::Magenta),
        ));
    }

    status_parts.push(Span::raw(
        " | Enter: Send | Shift+Enter: Newline | PgUp/PgDn: Scroll | Tab: Thinking | Ctrl+C: Quit ",
    ));

    let line = Line::from(status_parts);
    let paragraph = Paragraph::new(line).style(Style::default().bg(Color::Black));
    f.render_widget(paragraph, area);
}

pub fn render_model_list(
    f: &mut Frame,
    area: Rect,
    state: &crate::model_list::ModelListState,
    loaded_model: Option<&str>,
    loaded_backend: Option<&str>,
) {
    // Build title with loaded model info
    let title = if let Some(name) = loaded_model {
        format!(
            " Models (Enter: Load | Del: Delete | u: Unload) — Loaded: {} [{}] ",
            name,
            loaded_backend.unwrap_or("?")
        )
    } else {
        " Models (Enter: Load | Del: Delete) — No model loaded ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            title,
            Style::default()
                .fg(if loaded_model.is_some() {
                    Color::Green
                } else {
                    Color::Cyan
                })
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

    let loaded_name = loaded_model.unwrap_or("");

    let items: Vec<ListItem> = state
        .models
        .iter()
        .map(|m| {
            let is_loaded = m.name == loaded_name;
            let name_color = if is_loaded {
                Color::Green
            } else {
                Color::White
            };
            let mut spans = vec![Span::styled(
                m.name.clone(),
                Style::default().fg(name_color),
            )];
            if is_loaded {
                spans.push(Span::styled(" ● loaded", Style::default().fg(Color::Green)));
            }
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
    let tabs = ["Chat", "Models", "Browser", "Server", "Settings", "Logs"];
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

    // Show detected hardware at the top
    if let Some(ref hw) = state.hardware {
        lines.push(Line::styled(
            " Hardware Detected",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::styled(
            format!(
                "   CPU: {} threads | RAM: {}MB",
                hw.cpus, hw.memory_total_mb
            ),
            Style::default().fg(Color::Gray),
        ));
        if hw.gpus.is_empty() {
            lines.push(Line::styled(
                "   GPU: None detected (CPU-only mode)",
                Style::default().fg(Color::Yellow),
            ));
        } else {
            for gpu in &hw.gpus {
                lines.push(Line::styled(
                    format!(
                        "   GPU {}: {} ({}MB VRAM, {} used){}",
                        gpu.index,
                        gpu.name,
                        gpu.vram_total_mb,
                        gpu.vram_used_mb,
                        gpu.compute_capability
                            .as_ref()
                            .map(|c| format!(", CC {}", c))
                            .unwrap_or_default()
                    ),
                    Style::default().fg(Color::Magenta),
                ));
            }
            let runtimes: Vec<&str> = [
                ("CUDA", hw.has_cuda),
                ("ROCm", hw.has_rocm),
                ("Vulkan", hw.has_vulkan),
                ("Metal", hw.has_metal),
            ]
            .iter()
            .filter(|(_, ok)| *ok)
            .map(|(name, _)| *name)
            .collect();
            if !runtimes.is_empty() {
                lines.push(Line::styled(
                    format!("   Available runtimes: {}", runtimes.join(", ")),
                    Style::default().fg(Color::Cyan),
                ));
            }
        }
        lines.push(Line::from(""));
    }

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

pub fn render_logs(f: &mut Frame, area: Rect, state: &crate::log_buffer::LogsState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Live Logs — F6 ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    // Inner width of the log area (minus borders). Each log line is truncated
    // to this width so long lines (e.g. server startup banner) don't wrap and
    // break the layout.
    let inner_width = chunks[0].width.saturating_sub(2) as usize;

    let entries = state.entries();
    let mut lines: Vec<Line> = Vec::new();

    for entry in &entries {
        let level_color = match entry.level.as_str() {
            "ERROR" => Color::Red,
            "WARN" => Color::Yellow,
            "INFO" => Color::Green,
            "DEBUG" => Color::Blue,
            "TRACE" => Color::DarkGray,
            "LOG" => Color::Cyan,
            _ => Color::White,
        };

        // Fixed-width prefix: " HH:MM:SS.mmm LEVEL " = 14 + 6 = 20 chars
        let prefix_len = 20usize;
        // Truncate target to at most 25 chars to leave room for the message
        let max_target = 25usize;
        let target_display: String = if entry.target.chars().count() > max_target {
            entry.target.chars().take(max_target).collect()
        } else {
            entry.target.clone()
        };
        // target span + trailing space
        let target_span_len = target_display.chars().count() + 1;
        // Remaining width for the message
        let max_msg = inner_width.saturating_sub(prefix_len + target_span_len);

        let message_display = if entry.message.chars().count() > max_msg && max_msg > 3 {
            // Truncate with ellipsis
            let truncated: String = entry.message.chars().take(max_msg - 3).collect();
            format!("{}...", truncated)
        } else {
            entry.message.clone()
        };

        lines.push(Line::from(vec![
            Span::styled(
                format!(" {} ", entry.timestamp),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("{:<5} ", entry.level),
                Style::default()
                    .fg(level_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{} ", target_display),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(message_display, Style::default().fg(Color::White)),
        ]));
    }

    if lines.is_empty() {
        lines.push(Line::styled(
            " No logs yet. Logs will appear here in real-time.",
            Style::default().fg(Color::DarkGray),
        ));
    }

    // Auto-scroll to bottom, or use absolute scroll position.
    // Since we don't use Wrap, each Line is exactly one rendered row, so
    // lines.len() accurately reflects the number of visible rows.
    let total_lines = lines.len() as u16;
    let visible_height = chunks[0].height.saturating_sub(2);
    let max_scroll = total_lines.saturating_sub(visible_height);

    let scroll = if state.auto_scroll {
        max_scroll
    } else {
        // Use absolute scroll position. Clamp to valid range in case
        // entries were removed from the buffer (circular eviction).
        state.scroll_top.min(max_scroll)
    };

    let p = Paragraph::new(lines).block(block).scroll((scroll, 0));
    f.render_widget(p, chunks[0]);

    let status = if state.auto_scroll {
        " Auto-scroll: ON | ↑/↓: Scroll | PgUp/PgDn: Jump | End: Bottom | 'c': Clear | Esc: Back "
    } else {
        " Auto-scroll: OFF | ↑/↓: Scroll | PgUp/PgDn: Jump | End: Bottom | 'c': Clear | Esc: Back "
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
                " Enter to search | Ctrl+G: toggle GGUF filter | Ctrl+U: clear",
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
    // When server is running, reserve space for loaded models panel
    let loaded_count = if state.phase == ServerPhase::Running && !state.loaded_models.is_empty() {
        state.loaded_models.len()
    } else {
        0
    };
    // loaded models panel: header(1) + separator(1) + models(loaded_count) + endpoints(4) + padding(1) + borders(2)
    let loaded_panel_height = if loaded_count > 0 {
        (3 + loaded_count + 5) as u16
    } else {
        0
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),                   // Hardware info banner
            Constraint::Min(10),                     // Config fields (scrollable)
            Constraint::Length(loaded_panel_height), // Loaded models + endpoints
            Constraint::Length(3),                   // Status bar
        ])
        .split(area);

    render_hardware_banner(f, chunks[0], state);
    render_config_fields(f, chunks[1], state);
    if loaded_panel_height > 0 {
        render_loaded_models_panel(f, chunks[2], state);
    }
    render_server_status_bar(f, chunks[3], state);
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
        Span::styled("RAM: ", Style::default().fg(Color::Yellow)),
        Span::styled(
            format!(
                "{} free / {}",
                fmt_mb(state.hardware.memory_available_mb),
                fmt_mb(state.hardware.memory_total_mb),
            ),
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
            .map(|g| {
                format!(
                    "{} (VRAM {} free / {})",
                    g.name,
                    fmt_mb(g.vram_total_mb.saturating_sub(g.vram_used_mb)),
                    fmt_mb(g.vram_total_mb),
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    lines.push(Line::from(vec![
        Span::styled(" GPU: ", Style::default().fg(Color::Yellow)),
        Span::styled(gpu_str, Style::default().fg(Color::White)),
    ]));

    // Estimated memory footprint of the selected model — shown BEFORE
    // the user starts the server so they know whether it will fit.
    lines.push(render_load_estimate_line(state));

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

/// Build the "Est. load" banner line for the currently selected model.
fn render_load_estimate_line(state: &ServerPanelState) -> Line<'static> {
    let label = Span::styled(" Est. load: ", Style::default().fg(Color::Yellow));

    let Some(est) = state.estimate_selected_model_load() else {
        return Line::from(vec![
            label,
            Span::styled(
                "no model selected".to_string(),
                Style::default().fg(Color::DarkGray),
            ),
        ]);
    };

    let (verdict, verdict_color) = if !est.fits {
        ("✗ does NOT fit!", Color::Red)
    } else if est.tight {
        ("⚠ tight fit", Color::Yellow)
    } else {
        ("✓ fits", Color::Green)
    };

    let mut spans = vec![label];

    // What will be consumed
    if est.full_gpu_offload {
        spans.push(Span::styled(
            format!(
                "~{} VRAM + ~{} RAM",
                fmt_mb(est.vram_mb.unwrap_or(0)),
                fmt_mb(est.ram_mb),
            ),
            Style::default().fg(Color::White),
        ));
    } else if est.partial_gpu_offload {
        spans.push(Span::styled(
            format!(
                "≤{} RAM + ~{} VRAM ({} layers on GPU)",
                fmt_mb(est.ram_mb),
                fmt_mb(est.vram_mb.unwrap_or(0)),
                est.gpu_layers,
            ),
            Style::default().fg(Color::White),
        ));
    } else {
        spans.push(Span::styled(
            format!(
                "~{} RAM (model {} + ctx {})",
                fmt_mb(est.ram_mb),
                fmt_mb(est.model_size_mb),
                fmt_mb(est.ram_mb - est.model_size_mb),
            ),
            Style::default().fg(Color::White),
        ));
    }

    // What is available
    if est.ram_available_mb > 0 {
        spans.push(Span::styled(
            format!("  |  free: {}", fmt_mb(est.ram_available_mb)),
            Style::default().fg(Color::Gray),
        ));
    }
    if let Some(free) = est.vram_free_mb {
        if free > 0 {
            spans.push(Span::styled(
                format!(" + {} VRAM", fmt_mb(free)),
                Style::default().fg(Color::Gray),
            ));
        }
    }

    spans.push(Span::raw("  "));
    spans.push(Span::styled(
        verdict.to_string(),
        Style::default()
            .fg(verdict_color)
            .add_modifier(Modifier::BOLD),
    ));

    Line::from(spans)
}

/// Format a megabyte count as a human-readable string (GB when >= 1 GB).
fn fmt_mb(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{} MB", mb)
    }
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
    // Exact line index of the selected field — used for scroll math below
    let mut selected_line: u16 = 0;

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
        if is_selected {
            selected_line = lines.len() as u16;
        }
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
                format!("{}{: <22}: {}", prefix, field.label(), value),
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
                } else if state.phase == ServerPhase::LoadingModel {
                    (Color::Yellow, "■ Cancel Loading".to_string())
                } else {
                    (Color::DarkGray, "■ Server not running".to_string())
                }
            } else if *field == ConfigField::LoadAdditionalModel {
                if state.phase == ServerPhase::LoadingModel {
                    (Color::Cyan, "⟳ Loading model...".to_string())
                } else if state.phase == ServerPhase::Running {
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
            } else if *field == ConfigField::ManageApiKeys {
                if state.api_keys.is_empty() {
                    (Color::Cyan, "📋 API Keys (Enter to manage)".to_string())
                } else {
                    (
                        Color::Cyan,
                        format!("📋 {} key(s) (Enter to manage)", state.api_keys.len()),
                    )
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
                    ConfigField::ManageApiKeys => "Enter to open the API key management modal",
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
            Span::styled(format!("{}{: <22}: ", prefix, field.label()), style),
            Span::styled(value, value_style),
        ]));

        if is_selected && !state.editing {
            lines.push(Line::styled(
                format!("     hint: {}", state.field_hint(field)),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    // Keep the selected field (and its hint line) inside the visible area.
    // `selected_line` was recorded exactly while building the lines above,
    // so the scroll offset is precise — no estimation drift.
    let visible_height = area.height.saturating_sub(2); // minus borders
    let selected_end = selected_line + 1; // field line + hint line below it
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
}

fn render_loaded_models_panel(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            " Loaded Models & Endpoints ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    let mut lines: Vec<Line> = Vec::new();

    // Loaded models list
    for m in &state.loaded_models {
        let default_marker = if m.is_default { " ★ default" } else { "" };
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:>2}  ", m.id),
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

    // Endpoints
    lines.push(Line::from(""));
    lines.push(Line::styled(
        " Endpoints:",
        Style::default().fg(Color::DarkGray),
    ));
    lines.push(Line::styled(
        "   POST /v1/chat/completions   GET /v1/models",
        Style::default().fg(Color::Gray),
    ));
    lines.push(Line::styled(
        "   POST /v1/completions        GET /v1/health",
        Style::default().fg(Color::Gray),
    ));

    let p = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn render_server_status_bar(f: &mut Frame, area: Rect, state: &ServerPanelState) {
    let status = if state.editing {
        " Enter: Save | Esc: Cancel | Type to edit ".to_string()
    } else if let Some(ref msg) = state.status_message {
        // Multi-line messages (e.g. mmproj rejection) don't fit the bar —
        // show only the first line.
        msg.lines().next().unwrap_or(msg).to_string()
    } else if state.phase == ServerPhase::Running {
        format!(
            " Server running on {} | Edits apply on next start | Enter on Stop to halt ",
            state.server_url.as_deref().unwrap_or("?")
        )
    } else {
        " Up/Down: Navigate | PgUp/PgDn: Jump | Enter: Edit/Toggle/Action | Left/Right: Cycle Model | F6: Logs "
            .to_string()
    };

    let color = if state.status_message.is_some() && state.status_is_error {
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
