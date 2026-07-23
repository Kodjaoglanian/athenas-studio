use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{info, warn};

/// A single fallback chain entry — maps a primary model to a list of fallback models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackChain {
    /// The primary model id (or alias) that triggers this chain.
    pub primary: String,
    /// Ordered list of fallback model ids to try if the primary fails.
    pub fallbacks: Vec<String>,
    /// Maximum retries per model before moving to the next fallback.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Timeout in seconds per attempt (0 = no timeout).
    #[serde(default)]
    pub timeout_secs: u64,
}

fn default_max_retries() -> u32 {
    1
}

/// A routing alias — maps a virtual model name to a real model id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingAlias {
    pub alias: String,
    pub target: String,
}

/// The model router — manages routing aliases and fallback chains.
pub struct ModelRouter {
    /// Alias -> target model id mapping
    aliases: HashMap<String, String>,
    /// Primary model id -> fallback chain
    chains: HashMap<String, FallbackChain>,
    /// Health status: model_id -> (failure_count, last_failure_time)
    health: HashMap<String, ModelHealth>,
    /// Maximum failures before a model is marked unhealthy
    max_failures: u32,
    /// Cooldown period in seconds before retrying an unhealthy model
    cooldown_secs: u64,
}

#[derive(Debug, Clone, Default)]
struct ModelHealth {
    failures: u32,
    last_failure: Option<std::time::Instant>,
    healthy: bool,
}

impl ModelRouter {
    pub fn new() -> Self {
        Self {
            aliases: HashMap::new(),
            chains: HashMap::new(),
            health: HashMap::new(),
            max_failures: 3,
            cooldown_secs: 30,
        }
    }

    /// Create a router from config-defined chains and aliases.
    pub fn from_config(chains: Vec<FallbackChain>, aliases: Vec<RoutingAlias>) -> Self {
        let mut router = Self::new();
        for chain in chains {
            router.add_chain(chain);
        }
        for alias in aliases {
            router.add_alias(&alias.alias, &alias.target);
        }
        router
    }

    /// Add a routing alias.
    pub fn add_alias(&mut self, alias: &str, target: &str) {
        info!("Added routing alias: {} -> {}", alias, target);
        self.aliases.insert(alias.to_string(), target.to_string());
    }

    /// Remove a routing alias.
    pub fn remove_alias(&mut self, alias: &str) -> bool {
        self.aliases.remove(alias).is_some()
    }

    /// Add a fallback chain.
    pub fn add_chain(&mut self, chain: FallbackChain) {
        info!(
            "Added fallback chain: {} -> {:?}",
            chain.primary, chain.fallbacks
        );
        self.chains.insert(chain.primary.clone(), chain);
    }

    /// Remove a fallback chain.
    pub fn remove_chain(&mut self, primary: &str) -> bool {
        self.chains.remove(primary).is_some()
    }

    /// Resolve a model id (possibly an alias) to the actual model id.
    pub fn resolve(&self, model_id: &str) -> String {
        self.aliases
            .get(model_id)
            .cloned()
            .unwrap_or_else(|| model_id.to_string())
    }

    /// Get the fallback chain for a model, if any.
    pub fn get_chain(&self, model_id: &str) -> Option<&FallbackChain> {
        let resolved = self.resolve(model_id);
        self.chains.get(&resolved)
    }

    /// Get the ordered list of models to try for a given model id.
    /// This includes the primary model first, then fallbacks.
    pub fn get_model_sequence(&self, model_id: &str) -> Vec<String> {
        let resolved = self.resolve(model_id);
        let mut sequence = vec![resolved.clone()];
        if let Some(chain) = self.chains.get(&resolved) {
            for fallback in &chain.fallbacks {
                let fb_resolved = self.resolve(fallback);
                if !sequence.contains(&fb_resolved) {
                    sequence.push(fb_resolved);
                }
            }
        }
        sequence
    }

    /// Check if a model is currently healthy.
    pub fn is_healthy(&self, model_id: &str) -> bool {
        let resolved = self.resolve(model_id);
        if let Some(health) = self.health.get(&resolved) {
            if !health.healthy {
                // Check cooldown
                if let Some(last_fail) = health.last_failure {
                    let elapsed = last_fail.elapsed().as_secs();
                    if elapsed >= self.cooldown_secs {
                        return true; // Cooldown expired, allow retry
                    }
                    return false;
                }
            }
        }
        true
    }

    /// Record a successful request for a model.
    pub fn record_success(&mut self, model_id: &str) {
        let resolved = self.resolve(model_id);
        if let Some(health) = self.health.get_mut(&resolved) {
            health.failures = 0;
            health.healthy = true;
        }
    }

    /// Record a failure for a model. Returns true if the model should be skipped.
    pub fn record_failure(&mut self, model_id: &str) -> bool {
        let resolved = self.resolve(model_id);
        let health = self
            .health
            .entry(resolved.clone())
            .or_insert_with(|| ModelHealth {
                failures: 0,
                last_failure: None,
                healthy: true,
            });

        health.failures += 1;
        health.last_failure = Some(std::time::Instant::now());

        if health.failures >= self.max_failures {
            health.healthy = false;
            warn!(
                "Model '{}' marked unhealthy after {} failures",
                resolved, health.failures
            );
            return true;
        }
        false
    }

    /// Get health status for all models.
    pub fn health_status(&self) -> Vec<serde_json::Value> {
        self.health
            .iter()
            .map(|(model, h)| {
                serde_json::json!({
                    "model": model,
                    "healthy": h.healthy,
                    "failures": h.failures,
                })
            })
            .collect()
    }

    /// List all routing aliases.
    pub fn list_aliases(&self) -> Vec<(String, String)> {
        self.aliases
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// List all fallback chains.
    pub fn list_chains(&self) -> Vec<FallbackChain> {
        self.chains.values().cloned().collect()
    }

    /// Set the max failures threshold.
    pub fn set_max_failures(&mut self, max: u32) {
        self.max_failures = max;
    }

    /// Set the cooldown period in seconds.
    pub fn set_cooldown_secs(&mut self, secs: u64) {
        self.cooldown_secs = secs;
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

pub type SharedModelRouter = Arc<Mutex<ModelRouter>>;
