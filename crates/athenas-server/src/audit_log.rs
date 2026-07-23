use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// A single audit log entry recording an API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub endpoint: String,
    pub method: String,
    pub status: u16,
    pub key_id: Option<String>,
    pub key_name: Option<String>,
    pub model: Option<String>,
    pub tokens_prompt: u64,
    pub tokens_generated: u64,
    pub latency_ms: u64,
    pub client_ip: Option<String>,
    pub user_agent: Option<String>,
    pub error: Option<String>,
    pub request_id: Option<String>,
}

/// The audit logger — stores entries in a ring buffer and persists to disk.
pub struct AuditLogger {
    entries: VecDeque<AuditEntry>,
    max_entries: usize,
    data_dir: PathBuf,
    total_requests: u64,
    total_errors: u64,
}

impl AuditLogger {
    pub fn new(data_dir: PathBuf, max_entries: usize) -> Self {
        let mut logger = Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
            data_dir,
            total_requests: 0,
            total_errors: 0,
        };
        logger.load();
        logger
    }

    fn log_file(&self) -> PathBuf {
        self.data_dir.join("audit_log.jsonl")
    }

    fn load(&mut self) {
        if let Ok(data) = std::fs::read_to_string(self.log_file()) {
            for line in data.lines() {
                if line.is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<AuditEntry>(line) {
                    self.entries.push_back(entry);
                    if self.entries.len() > self.max_entries {
                        self.entries.pop_front();
                    }
                }
            }
            info!("Loaded {} audit log entries", self.entries.len());
        }
    }

    fn append_to_disk(&self, entry: &AuditEntry) {
        if let Some(parent) = self.log_file().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string(entry) {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.log_file())
            {
                if let Err(e) = writeln!(file, "{}", json) {
                    warn!("Failed to write audit log: {}", e);
                }
            }
        }
    }

    /// Record a new audit entry.
    pub fn record(&mut self, entry: AuditEntry) {
        self.total_requests += 1;
        if entry.status >= 400 {
            self.total_errors += 1;
        }
        self.append_to_disk(&entry);
        self.entries.push_back(entry);
        if self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
    }

    /// Query audit entries with optional filters.
    pub fn query(
        &self,
        limit: usize,
        key_id: Option<&str>,
        endpoint: Option<&str>,
        min_status: Option<u16>,
    ) -> Vec<AuditEntry> {
        let mut results: Vec<AuditEntry> = self
            .entries
            .iter()
            .rev()
            .filter(|e| {
                if let Some(kid) = key_id {
                    if e.key_id.as_deref() != Some(kid) {
                        return false;
                    }
                }
                if let Some(ep) = endpoint {
                    if !e.endpoint.contains(ep) {
                        return false;
                    }
                }
                if let Some(min) = min_status {
                    if e.status < min {
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .cloned()
            .collect();
        results.reverse();
        results
    }

    /// Get summary statistics.
    pub fn stats(&self) -> serde_json::Value {
        let total_tokens_prompt: u64 = self.entries.iter().map(|e| e.tokens_prompt).sum();
        let total_tokens_generated: u64 = self.entries.iter().map(|e| e.tokens_generated).sum();
        let avg_latency: u64 = if self.entries.is_empty() {
            0
        } else {
            self.entries.iter().map(|e| e.latency_ms).sum::<u64>() / self.entries.len() as u64
        };

        serde_json::json!({
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "entries_in_memory": self.entries.len(),
            "total_tokens_prompt": total_tokens_prompt,
            "total_tokens_generated": total_tokens_generated,
            "avg_latency_ms": avg_latency,
        })
    }

    /// Clear all in-memory entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

pub type SharedAuditLogger = Arc<Mutex<AuditLogger>>;
