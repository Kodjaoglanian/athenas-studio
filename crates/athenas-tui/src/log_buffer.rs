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
}

impl LogsState {
    pub fn new(buffer: LogBuffer) -> Self {
        Self {
            buffer,
            auto_scroll: true,
        }
    }

    pub fn entries(&self) -> Vec<LogEntry> {
        self.buffer.entries()
    }

    pub fn clear(&self) {
        self.buffer.clear();
    }
}
