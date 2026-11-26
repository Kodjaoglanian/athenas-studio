use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{debug, info};

/// A single cached entry — maps a query embedding to its response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The original user message (for debugging / display).
    pub query: String,
    /// The embedding of the query.
    pub embedding: Vec<f32>,
    /// The full chat completion response JSON (serialized).
    pub response_json: String,
    /// Model that generated the response.
    pub model: String,
    /// Hash of the system prompt(s) used for this request.
    #[serde(default)]
    pub system_hash: u64,
    /// Tenant / API key ID that owns this entry (prevents cross-tenant leakage).
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// Unix timestamp when the entry was created.
    pub created_at: u64,
    /// Number of tokens saved by this cache hit.
    pub tokens_saved: u64,
}

/// Cache statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub tokens_saved: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Configuration for the semantic cache — wraps the core config with a data_dir.
#[derive(Debug, Clone)]
pub struct SemanticCacheConfig {
    pub enabled: bool,
    /// Minimum cosine similarity to consider a hit (0.0 - 1.0).
    pub similarity_threshold: f32,
    /// Time-to-live for cache entries in seconds.
    pub ttl_secs: u64,
    /// Maximum number of entries (LRU eviction when exceeded).
    pub max_entries: usize,
    /// Data directory for persistence.
    pub data_dir: PathBuf,
}

impl From<athenas_core::SemanticCacheConfig> for SemanticCacheConfig {
    fn from(c: athenas_core::SemanticCacheConfig) -> Self {
        Self {
            enabled: c.enabled,
            similarity_threshold: c.similarity_threshold,
            ttl_secs: c.ttl_secs,
            max_entries: c.max_entries,
            // Expand ~ properly — PathBuf::from("~/...") is literal
            data_dir: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".athenas")
                .join("data"),
        }
    }
}

/// The semantic cache — stores query embeddings and their responses.
/// Uses cosine similarity to find matching queries.
pub struct SemanticCache {
    entries: HashMap<u64, CacheEntry>,
    /// LRU order: oldest keys first.
    lru_order: Vec<u64>,
    stats: CacheStats,
    config: SemanticCacheConfig,
}

impl SemanticCache {
    pub fn new(config: SemanticCacheConfig) -> Self {
        let mut cache = Self {
            entries: HashMap::new(),
            lru_order: Vec::new(),
            stats: CacheStats::default(),
            config,
        };
        cache.load();
        cache
    }

    fn cache_file(&self) -> PathBuf {
        self.config.data_dir.join("semantic_cache.json")
    }

    fn load(&mut self) {
        if let Ok(data) = std::fs::read_to_string(self.cache_file()) {
            if let Ok(entries) = serde_json::from_str::<Vec<CacheEntry>>(&data) {
                let now = current_timestamp();
                for entry in entries {
                    // Skip expired entries on load
                    if now.saturating_sub(entry.created_at) > self.config.ttl_secs {
                        continue;
                    }
                    let key = hash_cache_key(
                        &entry.embedding,
                        &entry.model,
                        entry.system_hash,
                        entry.tenant_id.as_deref(),
                    );
                    self.entries.insert(key, entry);
                    self.lru_order.push(key);
                }
                self.stats.entries = self.entries.len();
                info!("Loaded {} semantic cache entries", self.entries.len());
            }
        }
    }

    fn save(&self) {
        let path = self.cache_file();
        let entries: Vec<CacheEntry> = self.entries.values().cloned().collect();
        let Ok(json) = serde_json::to_string_pretty(&entries) else {
            return;
        };
        let write_fn = move || {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(&path, json) {
                debug!("Failed to save semantic cache: {}", e);
            }
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn_blocking(write_fn);
        } else {
            write_fn();
        }
    }

    /// Search for a matching entry by embedding similarity, scoped by model,
    /// system prompt hash, and tenant/API key ID to prevent cross-tenant leakage.
    pub fn lookup(
        &mut self,
        query_embedding: &[f32],
        model: &str,
        system_hash: u64,
        tenant_id: Option<&str>,
    ) -> Option<CacheEntry> {
        let now = current_timestamp();
        let query_hash = hash_cache_key(query_embedding, model, system_hash, tenant_id);

        // First try exact hash match (fast path)
        if let Some(entry) = self.entries.get(&query_hash) {
            if entry.model == model
                && entry.system_hash == system_hash
                && entry.tenant_id.as_deref() == tenant_id
            {
                if now.saturating_sub(entry.created_at) <= self.config.ttl_secs {
                    self.stats.hits += 1;
                    self.stats.tokens_saved += entry.tokens_saved;
                    let result = entry.clone();
                    // Move to end of LRU (most recently used)
                    self.lru_order.retain(|&k| k != query_hash);
                    self.lru_order.push(query_hash);
                    debug!("Semantic cache EXACT hit: {}", result.query);
                    return Some(result);
                } else {
                    // Expired — remove
                    self.entries.remove(&query_hash);
                    self.lru_order.retain(|&k| k != query_hash);
                }
            }
        }

        // Semantic search — find closest entry matching model, system prompt, and tenant
        let mut best: Option<(u64, f32, &CacheEntry)> = None;
        for (key, entry) in &self.entries {
            if entry.model != model
                || entry.system_hash != system_hash
                || entry.tenant_id.as_deref() != tenant_id
            {
                continue;
            }
            // Skip expired
            if now.saturating_sub(entry.created_at) > self.config.ttl_secs {
                continue;
            }
            let sim = cosine_similarity(query_embedding, &entry.embedding);
            if sim >= self.config.similarity_threshold && (best.is_none() || sim > best.unwrap().1)
            {
                best = Some((*key, sim, entry));
            }
        }

        if let Some((key, sim, entry)) = best {
            self.stats.hits += 1;
            self.stats.tokens_saved += entry.tokens_saved;
            debug!("Semantic cache hit (sim={:.3}): {}", sim, entry.query);
            let result = entry.clone();
            // Move to end of LRU (most recently used)
            self.lru_order.retain(|&k| k != key);
            self.lru_order.push(key);
            Some(result)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a new entry into the cache.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &mut self,
        query: String,
        query_embedding: Vec<f32>,
        response_json: String,
        model: String,
        system_hash: u64,
        tenant_id: Option<String>,
        tokens_saved: u64,
    ) {
        let key = hash_cache_key(&query_embedding, &model, system_hash, tenant_id.as_deref());
        let entry = CacheEntry {
            query,
            embedding: query_embedding,
            response_json,
            model,
            system_hash,
            tenant_id,
            created_at: current_timestamp(),
            tokens_saved,
        };

        // If key already exists, update it
        if self.entries.contains_key(&key) {
            self.lru_order.retain(|&k| k != key);
        }

        self.entries.insert(key, entry);
        self.lru_order.push(key);

        // Evict oldest entries if over capacity
        while self.lru_order.len() > self.config.max_entries {
            if let Some(oldest) = self.lru_order.first().copied() {
                self.entries.remove(&oldest);
                self.lru_order.remove(0);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }

        self.stats.entries = self.entries.len();
        self.save();
    }

    /// Get current cache stats.
    pub fn stats(&self) -> CacheStats {
        let mut stats = self.stats.clone();
        stats.entries = self.entries.len();
        stats
    }

    /// Clear all cache entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
        self.stats.entries = 0;
        self.save();
        info!("Semantic cache cleared");
    }

    /// Check if the cache is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Hash a system prompt string for cache scoping.
pub fn hash_system_prompt(system_prompt: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    system_prompt.hash(&mut hasher);
    hasher.finish()
}

/// Fast hash of an embedding vector + scope (model, system_hash, tenant_id) for exact-match lookup.
fn hash_cache_key(
    embedding: &[f32],
    model: &str,
    system_hash: u64,
    tenant_id: Option<&str>,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    model.hash(&mut hasher);
    system_hash.hash(&mut hasher);
    tenant_id.hash(&mut hasher);
    for &v in embedding {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Get current unix timestamp.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub type SharedSemanticCache = Arc<Mutex<SemanticCache>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a unique temp directory for each test to avoid cache file collisions.
    fn test_data_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "athenas_cache_test_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        dir
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_similar() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 2.9];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99);
    }

    #[test]
    fn test_cache_lookup_exact() {
        let config = SemanticCacheConfig {
            enabled: true,
            similarity_threshold: 0.9,
            ttl_secs: 3600,
            max_entries: 100,
            data_dir: test_data_dir("exact"),
        };
        let mut cache = SemanticCache::new(config);
        let embedding = vec![1.0, 2.0, 3.0];
        let sys_hash = hash_system_prompt("you are helpful");
        cache.insert(
            "hello".to_string(),
            embedding.clone(),
            "response".to_string(),
            "model".to_string(),
            sys_hash,
            Some("key_1".to_string()),
            100,
        );

        let result = cache.lookup(&embedding, "model", sys_hash, Some("key_1"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().response_json, "response");
        assert_eq!(cache.stats().hits, 1);

        // Cross-tenant, different model, and different system prompt must NOT hit
        assert!(cache
            .lookup(&embedding, "model", sys_hash, Some("key_2"))
            .is_none());
        assert!(cache
            .lookup(&embedding, "other-model", sys_hash, Some("key_1"))
            .is_none());
        assert!(cache
            .lookup(
                &embedding,
                "model",
                hash_system_prompt("different"),
                Some("key_1")
            )
            .is_none());
    }

    #[test]
    fn test_cache_lookup_similar() {
        let config = SemanticCacheConfig {
            enabled: true,
            similarity_threshold: 0.95,
            ttl_secs: 3600,
            max_entries: 100,
            data_dir: test_data_dir("similar"),
        };
        let mut cache = SemanticCache::new(config);
        cache.insert(
            "hello world".to_string(),
            vec![1.0, 2.0, 3.0],
            "response".to_string(),
            "model".to_string(),
            0,
            None,
            50,
        );

        // Similar but not identical
        let result = cache.lookup(&[1.01, 2.01, 2.99], "model", 0, None);
        assert!(result.is_some());
        // Different tenant must not hit even via similarity scan
        assert!(cache
            .lookup(&[1.01, 2.01, 2.99], "model", 0, Some("tenant_a"))
            .is_none());
    }

    #[test]
    fn test_cache_lookup_miss() {
        let config = SemanticCacheConfig {
            enabled: true,
            similarity_threshold: 0.99,
            ttl_secs: 3600,
            max_entries: 100,
            data_dir: test_data_dir("miss"),
        };
        let mut cache = SemanticCache::new(config);
        cache.insert(
            "hello".to_string(),
            vec![1.0, 2.0, 3.0],
            "response".to_string(),
            "model".to_string(),
            0,
            None,
            50,
        );

        // Very different
        let result = cache.lookup(&[0.0, 0.0, 1.0], "model", 0, None);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_eviction_and_exact_hit_lru() {
        let config = SemanticCacheConfig {
            enabled: true,
            similarity_threshold: 0.9,
            ttl_secs: 3600,
            max_entries: 3,
            data_dir: test_data_dir("eviction"),
        };
        let mut cache = SemanticCache::new(config);
        for i in 0..3 {
            cache.insert(
                format!("query {}", i),
                vec![(i + 1) as f32, 0.0, 0.0],
                format!("response {}", i),
                "model".to_string(),
                0,
                None,
                10,
            );
        }
        // Touch query 0 via exact hit so it becomes MRU
        assert!(cache.lookup(&[1.0, 0.0, 0.0], "model", 0, None).is_some());
        // Insert 2 more -> query 1 and query 2 should be evicted, query 0 retained
        for i in 3..5 {
            cache.insert(
                format!("query {}", i),
                vec![(i + 1) as f32, 0.0, 0.0],
                format!("response {}", i),
                "model".to_string(),
                0,
                None,
                10,
            );
        }
        assert_eq!(cache.entries.len(), 3);
        assert!(cache.stats().evictions >= 2);
        assert!(cache.lookup(&[1.0, 0.0, 0.0], "model", 0, None).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let config = SemanticCacheConfig {
            enabled: true,
            similarity_threshold: 0.9,
            ttl_secs: 3600,
            max_entries: 100,
            data_dir: test_data_dir("clear"),
        };
        let mut cache = SemanticCache::new(config);
        cache.insert(
            "hello".to_string(),
            vec![1.0, 2.0, 3.0],
            "response".to_string(),
            "model".to_string(),
            0,
            None,
            50,
        );
        assert_eq!(cache.entries.len(), 1);
        cache.clear();
        assert_eq!(cache.entries.len(), 0);
    }
}
