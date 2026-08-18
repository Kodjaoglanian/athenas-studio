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
            data_dir: PathBuf::from("~/.athenas/data"),
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
                    if now - entry.created_at > self.config.ttl_secs {
                        continue;
                    }
                    let key = hash_embedding(&entry.embedding);
                    self.entries.insert(key, entry);
                    self.lru_order.push(key);
                }
                self.stats.entries = self.entries.len();
                info!("Loaded {} semantic cache entries", self.entries.len());
            }
        }
    }

    fn save(&self) {
        if let Some(parent) = self.cache_file().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let entries: Vec<&CacheEntry> = self.entries.values().collect();
        if let Ok(json) = serde_json::to_string_pretty(&entries) {
            if let Err(e) = std::fs::write(self.cache_file(), json) {
                debug!("Failed to save semantic cache: {}", e);
            }
        }
    }

    /// Search for a matching entry by embedding similarity.
    /// Returns the cached response if a match is found above the threshold.
    pub fn lookup(&mut self, query_embedding: &[f32]) -> Option<CacheEntry> {
        let now = current_timestamp();
        let query_hash = hash_embedding(query_embedding);

        // First try exact hash match (fast path)
        if let Some(entry) = self.entries.get(&query_hash) {
            if now - entry.created_at <= self.config.ttl_secs {
                self.stats.hits += 1;
                self.stats.tokens_saved += entry.tokens_saved;
                debug!("Semantic cache EXACT hit: {}", entry.query);
                return Some(entry.clone());
            } else {
                // Expired — remove
                self.entries.remove(&query_hash);
                self.lru_order.retain(|&k| k != query_hash);
            }
        }

        // Semantic search — find closest entry
        let mut best: Option<(u64, f32, &CacheEntry)> = None;
        for (key, entry) in &self.entries {
            // Skip expired
            if now - entry.created_at > self.config.ttl_secs {
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
            // Move to end of LRU (most recently used)
            self.lru_order.retain(|&k| k != key);
            self.lru_order.push(key);
            Some(entry.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a new entry into the cache.
    pub fn insert(
        &mut self,
        query: String,
        query_embedding: Vec<f32>,
        response_json: String,
        model: String,
        tokens_saved: u64,
    ) {
        let key = hash_embedding(&query_embedding);
        let entry = CacheEntry {
            query,
            embedding: query_embedding,
            response_json,
            model,
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

/// Fast hash of an embedding vector for exact-match lookup.
fn hash_embedding(embedding: &[f32]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    // Hash the raw bytes of the f32 slice
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embedding.as_ptr() as *const u8,
            std::mem::size_of_val(embedding),
        )
    };
    bytes.hash(&mut hasher);
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
        cache.insert(
            "hello".to_string(),
            embedding.clone(),
            "response".to_string(),
            "model".to_string(),
            100,
        );

        let result = cache.lookup(&embedding);
        assert!(result.is_some());
        assert_eq!(result.unwrap().response_json, "response");
        assert_eq!(cache.stats().hits, 1);
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
            50,
        );

        // Similar but not identical
        let result = cache.lookup(&[1.01, 2.01, 2.99]);
        assert!(result.is_some());
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
            50,
        );

        // Very different
        let result = cache.lookup(&[0.0, 0.0, 1.0]);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let config = SemanticCacheConfig {
            enabled: true,
            similarity_threshold: 0.9,
            ttl_secs: 3600,
            max_entries: 3,
            data_dir: test_data_dir("eviction"),
        };
        let mut cache = SemanticCache::new(config);
        for i in 0..5 {
            cache.insert(
                format!("query {}", i),
                vec![i as f32, 0.0, 0.0],
                format!("response {}", i),
                "model".to_string(),
                10,
            );
        }
        assert_eq!(cache.entries.len(), 3);
        assert!(cache.stats().evictions >= 2);
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
            50,
        );
        assert_eq!(cache.entries.len(), 1);
        cache.clear();
        assert_eq!(cache.entries.len(), 0);
    }
}
