use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// A single API key with its associated metadata, quotas, and usage tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub key_id: String,
    pub api_key: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub active: bool,
    pub rate_limit_per_minute: u32,
    pub daily_token_limit: u64,
    pub allowed_models: Vec<String>,
    pub metadata: serde_json::Value,
}

/// Daily usage tracking for a single API key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyUsage {
    pub date: String,
    pub requests: u64,
    pub tokens_prompt: u64,
    pub tokens_generated: u64,
}

impl KeyUsage {
    pub fn tokens_total(&self) -> u64 {
        self.tokens_prompt + self.tokens_generated
    }
}

/// In-memory rate limit state for a key (token bucket per minute).
#[derive(Debug, Clone, Default)]
struct RateLimitState {
    tokens: f64,
    last_refill: Option<std::time::Instant>,
}

/// The API key manager — stores keys, validates requests, tracks usage.
pub struct ApiKeyManager {
    keys: HashMap<String, ApiKey>,
    usage: HashMap<String, KeyUsage>,
    rate_limit_states: HashMap<String, RateLimitState>,
    data_dir: PathBuf,
}

impl ApiKeyManager {
    pub fn new(data_dir: PathBuf) -> Self {
        let mut mgr = Self {
            keys: HashMap::new(),
            usage: HashMap::new(),
            rate_limit_states: HashMap::new(),
            data_dir,
        };
        mgr.load();
        mgr
    }

    fn keys_file(&self) -> PathBuf {
        self.data_dir.join("api_keys.json")
    }

    fn usage_file(&self) -> PathBuf {
        self.data_dir.join("key_usage.json")
    }

    fn load(&mut self) {
        if let Ok(data) = std::fs::read_to_string(self.keys_file()) {
            if let Ok(keys) = serde_json::from_str::<Vec<ApiKey>>(&data) {
                for key in keys {
                    self.keys.insert(key.api_key.clone(), key);
                }
                info!("Loaded {} API keys", self.keys.len());
            }
        }
        if let Ok(data) = std::fs::read_to_string(self.usage_file()) {
            if let Ok(usage_map) = serde_json::from_str::<HashMap<String, KeyUsage>>(&data) {
                self.usage = usage_map;
            }
        }
    }

    fn save_keys(&self) {
        if let Some(parent) = self.keys_file().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let keys: Vec<&ApiKey> = self.keys.values().collect();
        if let Ok(json) = serde_json::to_string_pretty(&keys) {
            if let Err(e) = std::fs::write(self.keys_file(), json) {
                warn!("Failed to save API keys: {}", e);
            }
        }
    }

    fn save_usage(&self) {
        if let Some(parent) = self.usage_file().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&self.usage) {
            if let Err(e) = std::fs::write(self.usage_file(), json) {
                warn!("Failed to save key usage: {}", e);
            }
        }
    }

    /// Create a new API key.
    pub fn create_key(
        &mut self,
        name: &str,
        rate_limit_per_minute: u32,
        daily_token_limit: u64,
        allowed_models: Vec<String>,
        metadata: Option<serde_json::Value>,
    ) -> ApiKey {
        let key_id = format!("key_{}", &uuid::Uuid::new_v4().to_string()[..8]);
        let api_key = format!("sk-ath-{}", uuid::Uuid::new_v4().simple());
        let key = ApiKey {
            key_id: key_id.clone(),
            api_key: api_key.clone(),
            name: name.to_string(),
            created_at: Utc::now(),
            expires_at: None,
            active: true,
            rate_limit_per_minute,
            daily_token_limit,
            allowed_models,
            metadata: metadata.unwrap_or(serde_json::Value::Null),
        };
        self.keys.insert(api_key.clone(), key.clone());
        self.save_keys();
        info!("Created API key: {} ({})", key_id, name);
        key
    }

    /// Revoke (deactivate) a key by key_id.
    pub fn revoke_key(&mut self, key_id: &str) -> bool {
        let found = self.keys.values_mut().find(|k| k.key_id == key_id);
        if let Some(key) = found {
            key.active = false;
            self.save_keys();
            info!("Revoked API key: {}", key_id);
            true
        } else {
            false
        }
    }

    /// Delete a key entirely.
    pub fn delete_key(&mut self, key_id: &str) -> bool {
        let api_key = self
            .keys
            .iter()
            .find(|(_, k)| k.key_id == key_id)
            .map(|(ak, _)| ak.clone());
        if let Some(ak) = api_key {
            self.keys.remove(&ak);
            self.usage.remove(&ak);
            self.rate_limit_states.remove(&ak);
            self.save_keys();
            self.save_usage();
            true
        } else {
            false
        }
    }

    /// List all keys (without exposing the actual api_key string).
    pub fn list_keys(&self) -> Vec<ApiKey> {
        self.keys.values().cloned().collect()
    }

    /// Get key info by key_id.
    pub fn get_key(&self, key_id: &str) -> Option<&ApiKey> {
        self.keys.values().find(|k| k.key_id == key_id)
    }

    /// Validate an API key string. Returns the key if valid and active.
    pub fn validate(&self, api_key: &str) -> Option<&ApiKey> {
        self.keys
            .get(api_key)
            .filter(|k| k.active && k.expires_at.map(|exp| exp > Utc::now()).unwrap_or(true))
    }

    /// Check if a key is allowed to use a specific model.
    pub fn check_model_access(&self, key: &ApiKey, model: &str) -> bool {
        if key.allowed_models.is_empty() {
            return true;
        }
        key.allowed_models.iter().any(|m| m == model)
    }

    /// Check and consume rate limit tokens for a key.
    /// Returns true if the request is allowed, false if rate limited.
    pub fn check_rate_limit(&mut self, key: &ApiKey) -> bool {
        if key.rate_limit_per_minute == 0 {
            return true;
        }
        let now = std::time::Instant::now();
        let state = self
            .rate_limit_states
            .entry(key.api_key.clone())
            .or_insert_with(|| RateLimitState {
                tokens: key.rate_limit_per_minute as f64,
                last_refill: Some(now),
            });

        if let Some(last) = state.last_refill {
            let elapsed = now.duration_since(last);
            let refill = elapsed.as_secs_f64() * (key.rate_limit_per_minute as f64 / 60.0);
            state.tokens = (state.tokens + refill).min(key.rate_limit_per_minute as f64);
        }
        state.last_refill = Some(now);

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Record token usage for a key.
    pub fn record_usage(&mut self, key: &ApiKey, tokens_prompt: u64, tokens_generated: u64) {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let usage = self
            .usage
            .entry(key.api_key.clone())
            .or_insert_with(|| KeyUsage {
                date: today.clone(),
                ..Default::default()
            });

        // Reset if it's a new day
        if usage.date != today {
            *usage = KeyUsage {
                date: today.clone(),
                ..Default::default()
            };
        }

        usage.requests += 1;
        usage.tokens_prompt += tokens_prompt;
        usage.tokens_generated += tokens_generated;
        self.save_usage();
    }

    /// Check if a key has exceeded its daily token limit.
    pub fn check_token_quota(&self, key: &ApiKey) -> bool {
        if key.daily_token_limit == 0 {
            return true;
        }
        let today = Utc::now().format("%Y-%m-%d").to_string();
        if let Some(usage) = self.usage.get(&key.api_key) {
            if usage.date == today {
                return usage.tokens_total() < key.daily_token_limit;
            }
        }
        true
    }

    /// Get usage stats for a key.
    pub fn get_usage(&self, key_id: &str) -> Option<(&ApiKey, KeyUsage)> {
        let key = self.keys.values().find(|k| k.key_id == key_id)?;
        let usage = self.usage.get(&key.api_key).cloned().unwrap_or_default();
        Some((key, usage))
    }

    /// Get remaining rate limit tokens for a key.
    pub fn rate_limit_remaining(&self, key: &ApiKey) -> u32 {
        if key.rate_limit_per_minute == 0 {
            return u32::MAX;
        }
        self.rate_limit_states
            .get(&key.api_key)
            .map(|s| s.tokens as u32)
            .unwrap_or(key.rate_limit_per_minute)
    }

    /// Persist all state to disk.
    pub fn save(&self) {
        self.save_keys();
        self.save_usage();
    }
}

pub type SharedApiKeyManager = Arc<Mutex<ApiKeyManager>>;

/// Result of validating an API key for a request.
pub enum AuthResult {
    /// No API key auth configured — request allowed.
    NoAuthRequired,
    /// Key is valid and request is allowed.
    Allowed { key_id: String, key_name: String },
    /// Key is invalid, expired, or revoked.
    Unauthorized,
    /// Key is valid but rate limited.
    RateLimited,
    /// Key is valid but token quota exceeded.
    QuotaExceeded,
    /// Key is valid but not allowed to use the requested model.
    Forbidden,
}
