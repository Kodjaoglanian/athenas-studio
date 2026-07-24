use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// A single document entry in the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Result of a similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document: VectorDocument,
    pub score: f32,
}

/// Configuration for the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Whether the vector store is enabled.
    pub enabled: bool,
    /// Directory for persisting the vector store data.
    pub data_dir: PathBuf,
    /// Maximum number of documents to store (0 = unlimited).
    pub max_documents: usize,
    /// Number of results to return by default in search.
    pub default_top_k: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            data_dir: PathBuf::new(),
            max_documents: 0,
            default_top_k: 5,
        }
    }
}

/// In-memory vector store with disk persistence.
/// Uses cosine similarity for search.
pub struct VectorStore {
    config: VectorStoreConfig,
    documents: RwLock<HashMap<String, VectorDocument>>,
    storage_path: PathBuf,
}

impl VectorStore {
    pub fn new(config: VectorStoreConfig) -> Self {
        let storage_path = config.data_dir.join("vector_store.json");
        let documents = RwLock::new(HashMap::new());

        let store = Self {
            config,
            documents,
            storage_path,
        };

        // Attempt to load existing data
        store.load_from_disk_blocking();
        store
    }

    fn load_from_disk_blocking(&self) {
        if !self.storage_path.exists() {
            return;
        }

        match std::fs::read_to_string(&self.storage_path) {
            Ok(content) => match serde_json::from_str::<Vec<VectorDocument>>(&content) {
                Ok(docs) => {
                    let mut map = self.documents.blocking_write();
                    for doc in docs {
                        map.insert(doc.id.clone(), doc);
                    }
                    info!("Loaded {} documents from vector store", map.len());
                }
                Err(e) => {
                    warn!("Failed to parse vector store data: {}", e);
                }
            },
            Err(e) => {
                warn!("Failed to read vector store file: {}", e);
            }
        }
    }

    async fn save_to_disk(&self) {
        let docs: Vec<VectorDocument> = {
            let map = self.documents.read().await;
            map.values().cloned().collect()
        };

        match serde_json::to_string_pretty(&docs) {
            Ok(content) => {
                if let Err(e) = std::fs::write(&self.storage_path, content) {
                    warn!("Failed to write vector store: {}", e);
                }
            }
            Err(e) => {
                warn!("Failed to serialize vector store: {}", e);
            }
        }
    }

    /// Add a document with its embedding to the store.
    pub async fn add_document(
        &self,
        id: String,
        content: String,
        embedding: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<VectorDocument, String> {
        // Check max documents limit
        if self.config.max_documents > 0 {
            let map = self.documents.read().await;
            if map.len() >= self.config.max_documents && !map.contains_key(&id) {
                return Err("Maximum document limit reached".to_string());
            }
        }

        let doc = VectorDocument {
            id: id.clone(),
            content,
            embedding,
            metadata,
            created_at: chrono::Utc::now(),
        };

        {
            let mut map = self.documents.write().await;
            map.insert(id, doc.clone());
        }

        self.save_to_disk().await;
        Ok(doc)
    }

    /// Add multiple documents in batch.
    pub async fn add_documents(
        &self,
        documents: Vec<(String, String, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<Vec<VectorDocument>, String> {
        let mut added = Vec::new();

        {
            let mut map = self.documents.write().await;
            for (id, content, embedding, metadata) in documents {
                // Check limit
                if self.config.max_documents > 0
                    && map.len() >= self.config.max_documents
                    && !map.contains_key(&id)
                {
                    warn!("Skipping document {} - max limit reached", id);
                    continue;
                }

                let doc = VectorDocument {
                    id: id.clone(),
                    content,
                    embedding,
                    metadata,
                    created_at: chrono::Utc::now(),
                };
                map.insert(id.clone(), doc.clone());
                added.push(doc);
            }
        }

        if !added.is_empty() {
            self.save_to_disk().await;
        }

        Ok(added)
    }

    /// Get a document by ID.
    pub async fn get_document(&self, id: &str) -> Option<VectorDocument> {
        let map = self.documents.read().await;
        map.get(id).cloned()
    }

    /// Delete a document by ID.
    pub async fn delete_document(&self, id: &str) -> bool {
        let removed = {
            let mut map = self.documents.write().await;
            map.remove(id).is_some()
        };

        if removed {
            self.save_to_disk().await;
        }

        removed
    }

    /// List all documents (without embeddings for brevity).
    pub async fn list_documents(&self) -> Vec<serde_json::Value> {
        let map = self.documents.read().await;
        map.values()
            .map(|d| {
                serde_json::json!({
                    "id": d.id,
                    "content": d.content,
                    "metadata": d.metadata,
                    "created_at": d.created_at,
                    "embedding_dim": d.embedding.len(),
                })
            })
            .collect()
    }

    /// Search for similar documents using cosine similarity.
    pub async fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let map = self.documents.read().await;

        let mut results: Vec<SearchResult> = map
            .values()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                SearchResult {
                    document: doc.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.into_iter().take(top_k).collect()
    }

    /// Clear all documents.
    pub async fn clear(&self) {
        {
            let mut map = self.documents.write().await;
            map.clear();
        }
        self.save_to_disk().await;
    }

    /// Get the number of documents in the store.
    pub async fn len(&self) -> usize {
        let map = self.documents.read().await;
        map.len()
    }

    /// Check if the store is empty.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

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

/// Shared vector store type.
pub type SharedVectorStore = Arc<VectorStore>;
