use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// Render markdown text into styled Lines for ratatui.
///
/// Supports:
/// - Headers: #, ##, ###, ####, #####, ######
/// - Bold: **text**
/// - Italic: *text*
/// - Inline code: `text`
/// - Code blocks: ``` ... ```
/// - Lists: - item, * item, 1. item
/// - Blockquotes: > text
/// - Horizontal rules: --- or ***
/// - Links: [text](url) — rendered as text (url)
///
/// Each line is wrapped to the given width.
pub fn render_markdown(text: &str, width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;

    for raw_line in text.lines() {
        // Handle code block fences
        if raw_line.trim_start().starts_with("```") {
            if in_code_block {
                // End of code block
                lines.push(Line::styled(
                    "  ──────────────────────────".to_string(),
                    Style::default().fg(Color::DarkGray),
                ));
                in_code_block = false;
            } else {
                // Start of code block
                let lang = raw_line.trim_start().trim_start_matches('`').trim();
                let label = if lang.is_empty() { "code" } else { lang };
                lines.push(Line::from(vec![
                    Span::styled("  ┌─ ".to_string(), Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        label.to_string(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " ──────────────────────".to_string(),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            // Render code block content with monospace-like styling
            for wrapped in wrap_text(raw_line, width.saturating_sub(2)) {
                lines.push(Line::styled(
                    format!("  {} ", wrapped),
                    Style::default().fg(Color::Cyan),
                ));
            }
            continue;
        }

        // Headers
        if let Some(rest) = raw_line.strip_prefix("###### ") {
            push_header(&mut lines, rest, 6, width);
        } else if let Some(rest) = raw_line.strip_prefix("##### ") {
            push_header(&mut lines, rest, 5, width);
        } else if let Some(rest) = raw_line.strip_prefix("#### ") {
            push_header(&mut lines, rest, 4, width);
        } else if let Some(rest) = raw_line.strip_prefix("### ") {
            push_header(&mut lines, rest, 3, width);
        } else if let Some(rest) = raw_line.strip_prefix("## ") {
            push_header(&mut lines, rest, 2, width);
        } else if let Some(rest) = raw_line.strip_prefix("# ") {
            push_header(&mut lines, rest, 1, width);
        } else if raw_line.trim() == "---" || raw_line.trim() == "***" {
            // Horizontal rule
            lines.push(Line::styled(
                "  ─────────────────────────────────────────────────────────".to_string(),
                Style::default().fg(Color::DarkGray),
            ));
        } else if raw_line.trim_start().starts_with("> ") {
            // Blockquote
            let content = raw_line.trim_start().strip_prefix("> ").unwrap_or("");
            let spans = parse_inline(content);
            for wrapped in wrap_spans(&spans, width.saturating_sub(4)) {
                let mut line_spans = vec![Span::styled(
                    "  │ ".to_string(),
                    Style::default().fg(Color::DarkGray),
                )];
                line_spans.extend(wrapped);
                lines.push(Line::from(line_spans));
            }
        } else if raw_line.trim_start().starts_with("- ") || raw_line.trim_start().starts_with("* ")
        {
            // Unordered list item
            let content = raw_line.trim_start()[2..].to_string();
            let spans = parse_inline(&content);
            for (i, wrapped) in wrap_spans(&spans, width.saturating_sub(4))
                .iter()
                .enumerate()
            {
                let prefix = if i == 0 { "  • " } else { "    " };
                let mut line_spans = vec![Span::styled(
                    prefix.to_string(),
                    Style::default().fg(Color::Cyan),
                )];
                line_spans.extend(wrapped.clone());
                lines.push(Line::from(line_spans));
            }
        } else if let Some(rest) = parse_ordered_list_item(raw_line) {
            // Ordered list item (1. 2. etc)
            let (num, content) = rest;
            let spans = parse_inline(&content);
            let prefix = format!("  {}. ", num);
            for (i, wrapped) in wrap_spans(&spans, width.saturating_sub(4))
                .iter()
                .enumerate()
            {
                let p = if i == 0 { prefix.as_str() } else { "    " };
                let mut line_spans = vec![Span::styled(
                    p.to_string(),
                    Style::default().fg(Color::Cyan),
                )];
                line_spans.extend(wrapped.clone());
                lines.push(Line::from(line_spans));
            }
        } else if raw_line.trim().is_empty() {
            // Empty line
            lines.push(Line::from(""));
        } else {
            // Regular paragraph text — parse inline formatting
            let spans = parse_inline(raw_line);
            for wrapped in wrap_spans(&spans, width.saturating_sub(2)) {
                let mut line_spans = vec![Span::raw("  ")];
                line_spans.extend(wrapped);
                lines.push(Line::from(line_spans));
            }
        }
    }

    // If code block was never closed, close it
    if in_code_block {
        lines.push(Line::styled(
            "  ──────────────────────────".to_string(),
            Style::default().fg(Color::DarkGray),
        ));
    }

    lines
}

fn push_header(lines: &mut Vec<Line<'static>>, text: &str, level: usize, width: usize) {
    let (color, style) = match level {
        1 => (Color::Cyan, Modifier::BOLD | Modifier::UNDERLINED),
        2 => (Color::Cyan, Modifier::BOLD),
        3 => (Color::Blue, Modifier::BOLD),
        4 => (Color::Blue, Modifier::empty()),
        5 => (Color::Gray, Modifier::BOLD),
        _ => (Color::Gray, Modifier::empty()),
    };
    let spans = parse_inline(text);
    let prefix = "  ";
    for wrapped in wrap_spans(&spans, width.saturating_sub(2)) {
        let mut line_spans = vec![Span::styled(
            prefix.to_string(),
            Style::default().fg(color).add_modifier(style),
        )];
        line_spans.extend(wrapped.into_iter().map(|s| {
            // Apply header style to each span
            let mut new_style = s.style;
            new_style = new_style.fg(color).add_modifier(style);
            Span::styled(s.content, new_style)
        }));
        lines.push(Line::from(line_spans));
    }
}

/// Parse a line number prefix from an ordered list item.
/// Returns (number, content) if the line starts with "N. " where N is a number.
fn parse_ordered_list_item(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let dot_pos = trimmed.find(". ")?;
    let num_str = &trimmed[..dot_pos];
    let num: usize = num_str.parse().ok()?;
    let content = trimmed[dot_pos + 2..].to_string();
    Some((num, content))
}

/// Parse inline markdown formatting (bold, italic, code, links) into Spans.
fn parse_inline(text: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Inline code: `text`
        if chars[i] == '`' {
            if !current.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut current)));
            }
            let start = i + 1;
            let mut end = start;
            while end < chars.len() && chars[end] != '`' {
                end += 1;
            }
            if end < chars.len() {
                let code: String = chars[start..end].iter().collect();
                spans.push(Span::styled(
                    code,
                    Style::default().fg(Color::Cyan).bg(Color::Black),
                ));
                i = end + 1;
                continue;
            }
        }

        // Bold: **text**
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            if !current.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut current)));
            }
            let start = i + 2;
            let mut end = start;
            while end + 1 < chars.len() && !(chars[end] == '*' && chars[end + 1] == '*') {
                end += 1;
            }
            if end + 1 < chars.len() {
                let bold: String = chars[start..end].iter().collect();
                // Recursively parse inline within bold
                let inner = parse_inline(&bold);
                for s in inner {
                    spans.push(Span::styled(
                        s.content,
                        s.style.add_modifier(Modifier::BOLD),
                    ));
                }
                i = end + 2;
                continue;
            }
        }

        // Italic: *text* (but not ** which is bold)
        if chars[i] == '*' && (i + 1 >= chars.len() || chars[i + 1] != '*') {
            if !current.is_empty() {
                spans.push(Span::raw(std::mem::take(&mut current)));
            }
            let start = i + 1;
            let mut end = start;
            while end < chars.len() && chars[end] != '*' {
                end += 1;
            }
            if end < chars.len() {
                let italic: String = chars[start..end].iter().collect();
                spans.push(Span::styled(
                    italic,
                    Style::default().add_modifier(Modifier::ITALIC),
                ));
                i = end + 1;
                continue;
            }
        }

        // Links: [text](url)
        if chars[i] == '[' {
            let mut close_bracket = i + 1;
            while close_bracket < chars.len() && chars[close_bracket] != ']' {
                close_bracket += 1;
            }
            if close_bracket < chars.len()
                && close_bracket + 1 < chars.len()
                && chars[close_bracket + 1] == '('
            {
                let mut close_paren = close_bracket + 2;
                while close_paren < chars.len() && chars[close_paren] != ')' {
                    close_paren += 1;
                }
                if close_paren < chars.len() {
                    if !current.is_empty() {
                        spans.push(Span::raw(std::mem::take(&mut current)));
                    }
                    let link_text: String = chars[i + 1..close_bracket].iter().collect();
                    let url: String = chars[close_bracket + 2..close_paren].iter().collect();
                    spans.push(Span::styled(
                        link_text,
                        Style::default()
                            .fg(Color::Blue)
                            .add_modifier(Modifier::UNDERLINED),
                    ));
                    spans.push(Span::styled(
                        format!(" ({})", url),
                        Style::default().fg(Color::DarkGray),
                    ));
                    i = close_paren + 1;
                    continue;
                }
            }
        }

        current.push(chars[i]);
        i += 1;
    }

    if !current.is_empty() {
        spans.push(Span::raw(current));
    }

    spans
}

/// Word-wrap a plain text string to the given width.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut result: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;

    for word in text.split_whitespace() {
        let word_len = word.chars().count();
        if current_len == 0 {
            if word_len <= width {
                current = word.to_string();
                current_len = word_len;
            } else {
                let chars: Vec<char> = word.chars().collect();
                let mut idx = 0;
                while idx < chars.len() {
                    let end = (idx + width).min(chars.len());
                    let chunk: String = chars[idx..end].iter().collect();
                    result.push(chunk);
                    idx = end;
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
                let chars: Vec<char> = word.chars().collect();
                let mut idx = 0;
                while idx < chars.len() {
                    let end = (idx + width).min(chars.len());
                    let chunk: String = chars[idx..end].iter().collect();
                    result.push(chunk);
                    idx = end;
                }
            }
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    if result.is_empty() {
        result.push(String::new());
    }
    result
}

/// Word-wrap a sequence of Spans to the given width.
/// Returns a Vec of Vec<Span>, where each inner Vec is one wrapped line.
fn wrap_spans(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    if width == 0 {
        return vec![spans.to_vec()];
    }

    // Flatten spans into (char, style) pairs
    let mut chars: Vec<(char, Style)> = Vec::new();
    for span in spans {
        for c in span.content.chars() {
            chars.push((c, span.style));
        }
    }

    // Split into words (by spaces), keeping track of styles
    let mut words: Vec<(String, Vec<Style>)> = Vec::new();
    let mut current_word = String::new();
    let mut current_styles: Vec<Style> = Vec::new();

    for (c, style) in chars {
        if c == ' ' {
            if !current_word.is_empty() {
                words.push((
                    std::mem::take(&mut current_word),
                    std::mem::take(&mut current_styles),
                ));
            }
        } else {
            current_word.push(c);
            current_styles.push(style);
        }
    }
    if !current_word.is_empty() {
        words.push((current_word, current_styles));
    }

    // Build lines from words
    let mut lines: Vec<Vec<Span<'static>>> = Vec::new();
    let mut current_line: Vec<Span<'static>> = Vec::new();
    let mut current_len = 0usize;

    for (word, styles) in words {
        let word_len = word.chars().count();
        if current_len == 0 {
            // First word on line
            if word_len <= width {
                current_line.push(Span::styled(
                    word,
                    styles.first().copied().unwrap_or_default(),
                ));
                current_len = word_len;
            } else {
                // Word longer than width — hard break
                let mut idx = 0;
                let chars: Vec<char> = word.chars().collect();
                while idx < chars.len() {
                    let end = (idx + width).min(chars.len());
                    let chunk: String = chars[idx..end].iter().collect();
                    let chunk_style = styles.get(idx).copied().unwrap_or_default();
                    lines.push(vec![Span::styled(chunk, chunk_style)]);
                    idx = end;
                }
            }
        } else if current_len + 1 + word_len <= width {
            current_line.push(Span::raw(" "));
            current_line.push(Span::styled(
                word,
                styles.first().copied().unwrap_or_default(),
            ));
            current_len += 1 + word_len;
        } else {
            // Flush current line
            lines.push(std::mem::take(&mut current_line));
            current_len = 0;
            if word_len <= width {
                current_line.push(Span::styled(
                    word,
                    styles.first().copied().unwrap_or_default(),
                ));
                current_len = word_len;
            } else {
                let chars: Vec<char> = word.chars().collect();
                let mut idx = 0;
                while idx < chars.len() {
                    let end = (idx + width).min(chars.len());
                    let chunk: String = chars[idx..end].iter().collect();
                    let chunk_style = styles.get(idx).copied().unwrap_or_default();
                    lines.push(vec![Span::styled(chunk, chunk_style)]);
                    idx = end;
                }
            }
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }
    if lines.is_empty() {
        lines.push(vec![]);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_inline_plain() {
        let spans = parse_inline("hello world");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content, "hello world");
    }

    #[test]
    fn test_parse_inline_bold() {
        let spans = parse_inline("hello **world** foo");
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].content, "hello ");
        assert_eq!(spans[1].content, "world");
        assert!(spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(spans[2].content, " foo");
    }

    #[test]
    fn test_parse_inline_italic() {
        let spans = parse_inline("hello *world* foo");
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].content, "world");
        assert!(spans[1].style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn test_parse_inline_code() {
        let spans = parse_inline("use `cargo` now");
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].content, "cargo");
        assert_eq!(spans[1].style.fg, Some(Color::Cyan));
    }

    #[test]
    fn test_parse_inline_link() {
        let spans = parse_inline("see [docs](https://example.com) here");
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[1].content, "docs");
        assert!(spans[1].style.add_modifier.contains(Modifier::UNDERLINED));
        assert_eq!(spans[2].content, " (https://example.com)");
    }

    #[test]
    fn test_render_markdown_header() {
        let lines = render_markdown("# Title\nSome text", 80);
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_render_markdown_code_block() {
        let input = "```rust\nfn main() {}\n```";
        let lines = render_markdown(input, 80);
        // Should have: start border, code line, end border
        assert!(lines.len() >= 3);
    }

    #[test]
    fn test_render_markdown_list() {
        let input = "- item 1\n- item 2\n- item 3";
        let lines = render_markdown(input, 80);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_render_markdown_bold_italic() {
        let lines = render_markdown("This is **bold** and *italic*", 80);
        assert!(!lines.is_empty());
    }
}
