use ratatui::{
    layout::{Margin, Rect},
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, BorderType, Borders, Scrollbar, ScrollbarOrientation, ScrollbarState},
    Frame,
};

// Modern dark studio palette (Tailwind Slate / Sky / Indigo / Emerald / Amber / Rose)
pub const BG_HEADER: Color = Color::Rgb(15, 23, 42);
pub const BG_SURFACE: Color = Color::Rgb(22, 30, 46);
pub const BG_HIGHLIGHT: Color = Color::Rgb(30, 41, 59);
pub const BG_CODE: Color = Color::Rgb(17, 24, 39);

pub const BORDER: Color = Color::Rgb(71, 85, 105);
pub const BORDER_FOCUS: Color = Color::Rgb(56, 189, 248);

pub const PRIMARY: Color = Color::Rgb(56, 189, 248); // Sky-400
pub const SECONDARY: Color = Color::Rgb(129, 140, 248); // Indigo-400
pub const REASONING: Color = Color::Rgb(192, 132, 252); // Purple-400
pub const SUCCESS: Color = Color::Rgb(52, 211, 153); // Emerald-400
pub const WARNING: Color = Color::Rgb(251, 191, 36); // Amber-400
pub const ERROR: Color = Color::Rgb(251, 113, 133); // Rose-400

pub const TEXT: Color = Color::Rgb(241, 245, 249); // Slate-100
pub const TEXT_DIM: Color = Color::Rgb(148, 163, 184); // Slate-400
pub const TEXT_MUTED: Color = Color::Rgb(100, 116, 139); // Slate-500

const SPINNER_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn spinner_frame(tick: usize) -> &'static str {
    SPINNER_FRAMES[tick % SPINNER_FRAMES.len()]
}

pub fn rounded_block(title: impl Into<String>, focused: bool) -> Block<'static> {
    let border_color = if focused { BORDER_FOCUS } else { BORDER };
    let title_color = if focused { PRIMARY } else { TEXT_DIM };
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(Span::styled(
            title.into(),
            Style::default()
                .fg(title_color)
                .add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(border_color))
}

pub fn rounded_block_colored(title: impl Into<String>, color: Color) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(Span::styled(
            title.into(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ))
        .border_style(Style::default().fg(color))
}

pub fn pill(text: impl Into<String>, fg: Color, bg: Color) -> Span<'static> {
    Span::styled(
        format!(" {} ", text.into()),
        Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD),
    )
}

pub fn badge(text: impl Into<String>, color: Color) -> Span<'static> {
    Span::styled(
        format!("[{}]", text.into()),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

pub fn keycap(key: &str, label: &str) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            format!(" {} ", key),
            Style::default()
                .fg(PRIMARY)
                .bg(BG_HIGHLIGHT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {} ", label), Style::default().fg(TEXT_DIM)),
    ]
}

pub fn render_scrollbar(f: &mut Frame, area: Rect, total: usize, position: usize) {
    if total <= 1 || area.height < 3 {
        return;
    }
    let mut sb_state = ScrollbarState::new(total).position(position.min(total.saturating_sub(1)));
    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .thumb_symbol("┃")
        .track_symbol(Some("│"))
        .begin_symbol(None)
        .end_symbol(None)
        .thumb_style(Style::default().fg(PRIMARY))
        .track_style(Style::default().fg(BG_HIGHLIGHT));
    f.render_stateful_widget(
        scrollbar,
        area.inner(Margin {
            vertical: 1,
            horizontal: 0,
        }),
        &mut sb_state,
    );
}

/// Format bytes using binary units (GiB / MiB / KiB / B).
pub fn format_bytes_binary(bytes: u64) -> String {
    const GIB: f64 = 1_073_741_824.0;
    const MIB: f64 = 1_048_576.0;
    const KIB: f64 = 1024.0;

    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes as f64 >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB)
    } else if bytes as f64 >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB)
    } else {
        format!("{} B", bytes)
    }
}

/// Format megabytes as human-readable GiB/MiB.
pub fn fmt_mb(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GiB", mb as f64 / 1024.0)
    } else {
        format!("{} MiB", mb)
    }
}
