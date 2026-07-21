use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::chat::ChatState;

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
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
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
            Style::default().fg(Color::Cyan).add_modifier(Modifier::ITALIC),
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
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(Color::DarkGray));

    if state.models.is_empty() {
        let p = Paragraph::new("No models downloaded.\nUse 'athenas models pull <repo-id>' to download.")
            .block(block)
            .style(Style::default().fg(Color::Gray));
        f.render_widget(p, area);
        return;
    }

    let items: Vec<ListItem> = state
        .models
        .iter()
        .map(|m| {
            let mut spans = vec![
                Span::styled(m.name.clone(), Style::default().fg(Color::White)),
            ];
            if let Some(ref q) = m.quantization {
                spans.push(Span::styled(format!(" [{}]", q), Style::default().fg(Color::Yellow)));
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
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .highlight_symbol("> ");

    f.render_stateful_widget(list, area, &mut state.list_state.clone());
}

pub fn render_sidebar(f: &mut Frame, area: Rect, conversations: &[(String, String)], selected: usize) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Conversations ")
        .border_style(Style::default().fg(Color::DarkGray));

    let items: Vec<ListItem> = conversations
        .iter()
        .map(|(_id, title)| {
            ListItem::new(Line::from(title.as_str()))
        })
        .collect();

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .highlight_symbol("> ");

    let mut state = ratatui::widgets::ListState::default();
    state.select(Some(selected));
    f.render_stateful_widget(list, area, &mut state);
}
