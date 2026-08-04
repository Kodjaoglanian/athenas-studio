use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use chrono::Local;
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::Layer;

/// A single log entry displayed in the TUI logs page.
#[derive(Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub target: String,
    pub message: String,
}

/// Thread-safe circular buffer for log entries.
///
/// The buffer is purely in-memory — it is never persisted to disk. When the
/// TUI exits, all buffered logs are discarded. This keeps the host machine
/// clean while still showing real-time logs (including server requests) in
/// the logs page.
#[derive(Clone)]
pub struct LogBuffer {
    inner: Arc<RwLock<VecDeque<LogEntry>>>,
    max_entries: usize,
}

impl LogBuffer {
    pub fn new(max_entries: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(VecDeque::with_capacity(max_entries))),
            max_entries,
        }
    }

    pub fn push(&self, entry: LogEntry) {
        if let Ok(mut buf) = self.inner.write() {
            if buf.len() >= self.max_entries {
                buf.pop_front();
            }
            buf.push_back(entry);
        }
    }

    /// Push a raw log line (e.g. from the detached server's log file) into
    /// the buffer. Only tracing-formatted lines (starting with an RFC3339
    /// timestamp) are accepted. `println!` output (server banner, "Loading
    /// model:", etc.) and box-drawing lines are silently skipped.
    pub fn push_raw_line(&self, line: &str) {
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        // Strip ANSI escape codes (the server's tracing subscriber may emit
        // them even to a file if with_ansi wasn't disabled at the time)
        let clean = strip_ansi(line);
        if clean.is_empty() {
            return;
        }
        // Only accept lines that look like tracing output: they start with
        // an RFC3339 timestamp (contains 'T' and ends with 'Z' or has tz).
        if !is_tracing_log_line(&clean) {
            return;
        }
        let entry = parse_server_log_line(&clean);
        self.push(entry);
    }

    pub fn entries(&self) -> Vec<LogEntry> {
        self.inner
            .read()
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn clear(&self) {
        if let Ok(mut buf) = self.inner.write() {
            buf.clear();
        }
    }
}

/// Strip ANSI escape codes from a string.
fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // ESC sequence: skip until we find a letter (the final byte)
            // Common patterns: \x1b[...m, \x1b[...H, etc.
            // Skip the '[' or other intermediate byte
            if chars.peek() == Some(&'[') {
                chars.next();
            }
            // Skip until we find a letter in 0x40-0x7E range
            for inner in chars.by_ref() {
                if inner.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Check if a line looks like a tracing log line. Tracing `fmt` output
/// starts with an RFC3339 timestamp like "2024-08-04T16:33:43.768Z".
/// `println!` output (server banner, "Loading model:", etc.) does not.
fn is_tracing_log_line(line: &str) -> bool {
    // The timestamp is the first token (up to first space)
    let ts_end = line.find(' ').unwrap_or(line.len());
    let timestamp = &line[..ts_end];
    // RFC3339 timestamps contain 'T' (date/time separator)
    if !timestamp.contains('T') {
        return false;
    }
    // And end with 'Z' (UTC) or contain a timezone offset (+/-)
    timestamp.ends_with('Z') || timestamp.contains('+') || timestamp.contains('-')
}

/// Parse a tracing `fmt` log line into a LogEntry.
///
/// Expected format (after ANSI stripping):
/// ```text
/// 2024-08-04T16:33:43.768Z  INFO athenas_server::middleware: GET /v1/health 200 1ms from 127.0.0.1
/// ```
///
/// The level is right-aligned to 5 chars by tracing_subscriber, so "INFO"
/// appears as " INFO" (with a leading space). After trim_start on the
/// remainder, we get the level without padding.
fn parse_server_log_line(line: &str) -> LogEntry {
    // <timestamp>  <LEVEL> <target>: <message>
    let ts_end = line.find(' ').unwrap_or(line.len());
    let timestamp_raw = &line[..ts_end];

    // Skip spaces after timestamp (there are 2+ spaces due to level padding)
    let rest = line[ts_end..].trim_start();

    // Level is the first token (up to next space)
    let level_end = rest.find(' ').unwrap_or(rest.len());
    let level = rest[..level_end].trim().to_string();

    let rest = rest[level_end..].trim_start();

    // Target is everything up to ": "
    let (target, message) = if let Some(colon_pos) = rest.find(": ") {
        (
            rest[..colon_pos].to_string(),
            rest[colon_pos + 2..].to_string(),
        )
    } else {
        ("server".to_string(), rest.to_string())
    };

    // Convert the RFC3339 timestamp to a shorter HH:MM:SS.mmm format for display
    let timestamp = format_rfc3339_to_short(timestamp_raw);

    LogEntry {
        timestamp,
        level: if level.is_empty() {
            "INFO".to_string()
        } else {
            level
        },
        target,
        message,
    }
}

/// Convert an RFC3339 timestamp (e.g. "2024-01-15T10:30:45.123456Z") to a
/// short "HH:MM:SS.mmm" format. Falls back to the original string on failure.
fn format_rfc3339_to_short(ts: &str) -> String {
    // Extract the time portion after 'T'
    let time_part = match ts.find('T').map(|i| &ts[i + 1..]) {
        Some(t) => t,
        None => return ts.to_string(),
    };

    // Remove trailing 'Z' or timezone offset
    let time_part = time_part.trim_end_matches('Z');
    let time_part = match time_part.find(['+', '-']) {
        Some(i) => &time_part[..i],
        None => time_part,
    };

    // Split into seconds and fractional parts
    if let Some(dot_pos) = time_part.find('.') {
        let secs = &time_part[..dot_pos];
        let frac = &time_part[dot_pos + 1..];
        // Take first 3 digits of fractional part for milliseconds
        let ms = if frac.len() >= 3 { &frac[..3] } else { frac };
        format!("{}.{}", secs, ms)
    } else {
        time_part.to_string()
    }
}

/// A tracing layer that writes formatted log lines into a `LogBuffer`.
pub struct LogBufferLayer {
    buffer: LogBuffer,
}

impl LogBufferLayer {
    pub fn new(buffer: LogBuffer) -> Self {
        Self { buffer }
    }
}

impl<S> Layer<S> for LogBufferLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        // Format the event message
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let message = visitor.message.unwrap_or_default();
        let level = event.metadata().level().as_str().to_string();
        let target = event.metadata().target().to_string();
        let timestamp = Local::now().format("%H:%M:%S%.3f").to_string();

        self.buffer.push(LogEntry {
            timestamp,
            level,
            target,
            message,
        });
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: Option<String>,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else if self.message.is_none() {
            let val = format!("{}={}", field.name(), value);
            self.message = Some(val);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{:?}", value));
        } else if self.message.is_none() {
            let val = format!("{}={:?}", field.name(), value);
            match &mut self.message {
                Some(msg) => {
                    msg.push_str(&format!(" {}", val));
                }
                None => {
                    self.message = Some(val);
                }
            }
        }
    }
}

/// State for the TUI logs page.
pub struct LogsState {
    pub buffer: LogBuffer,
    pub auto_scroll: bool,
    /// Absolute scroll position — the index of the first visible line from
    /// the top of the log entries. Only used when `auto_scroll` is false.
    /// When new logs arrive while scrolled up, this value doesn't change,
    /// so the view stays frozen at the position the user chose.
    pub scroll_top: u16,
}

impl LogsState {
    pub fn new(buffer: LogBuffer) -> Self {
        Self {
            buffer,
            auto_scroll: true,
            scroll_top: 0,
        }
    }

    pub fn entries(&self) -> Vec<LogEntry> {
        self.buffer.entries()
    }

    pub fn clear(&self) {
        self.buffer.clear();
    }

    /// Scroll up (toward older logs) by `n` lines.
    /// Sets auto_scroll to false and moves the view up by decrementing
    /// scroll_top. The view will stay frozen at this position even as new
    /// logs arrive.
    pub fn scroll_up(&mut self, n: u16) {
        if self.auto_scroll {
            // First time scrolling up from auto-scroll: start from the
            // bottom and move up by n.
            self.auto_scroll = false;
            let total = self.buffer.entries().len() as u16;
            let visible = 0u16; // will be clamped in render
            let _ = visible;
            self.scroll_top = total.saturating_sub(n + 1);
        } else {
            self.scroll_top = self.scroll_top.saturating_sub(n);
        }
    }

    /// Scroll down (toward newer logs) by `n` lines.
    /// If we reach the bottom, re-enable auto-scroll.
    pub fn scroll_down(&mut self, n: u16) {
        if self.auto_scroll {
            return;
        }
        self.scroll_top = self.scroll_top.saturating_add(n);
        // Check if we've reached the bottom — if so, re-enable auto-scroll.
        // The exact threshold is checked in render, but we approximate here.
        let total = self.buffer.entries().len() as u16;
        let max_scroll = total.saturating_sub(1);
        if self.scroll_top >= max_scroll {
            self.auto_scroll = true;
            self.scroll_top = 0;
        }
    }

    /// Reset scroll to bottom and re-enable auto-scroll.
    pub fn scroll_to_bottom(&mut self) {
        self.scroll_top = 0;
        self.auto_scroll = true;
    }
}
