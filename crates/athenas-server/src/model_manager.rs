use std::collections::HashMap;
use std::sync::Arc;

use athenas_inference::{Backend, ModelInfo};
use tokio::sync::Mutex;

/// Shared backend reference that can outlive the manager lock.
pub type SharedBackend = Arc<dyn Backend>;

/// Information about a loaded model entry in the manager.
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub id: String,
    pub model_info: ModelInfo,
    pub backend_name: String,
}

/// Manages multiple loaded model backends, keyed by model name.
/// Supports loading, unloading, and routing requests to the correct backend.
pub struct ModelManager {
    /// Map of model_id -> backend
    backends: HashMap<String, SharedBackend>,
    /// The default model id used when no model is specified in a request
    default_model: Option<String>,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            default_model: None,
        }
    }

    /// Create a manager pre-loaded with a single backend (backward compat).
    pub fn with_default(backend: Box<dyn Backend>) -> Self {
        let mut mgr = Self::new();
        if let Some(info) = backend.model_info() {
            let id = info.name.clone();
            mgr.backends.insert(id.clone(), Arc::from(backend));
            mgr.default_model = Some(id);
        }
        mgr
    }

    /// Add a backend to the manager. If it's the first, it becomes the default.
    pub fn add(&mut self, backend: Box<dyn Backend>) -> String {
        let info = backend.model_info().unwrap_or(ModelInfo {
            name: "unknown".to_string(),
            context_size: 0,
            gpu_layers: 0,
            backend_name: backend.name().to_string(),
        });
        let id = info.name.clone();
        self.backends.insert(id.clone(), Arc::from(backend));
        if self.default_model.is_none() {
            self.default_model = Some(id.clone());
        }
        id
    }

    /// Remove a model from the manager.
    ///
    /// The backend is immediately detached from the manager so no new requests
    /// are routed to it. If in-flight inference requests still hold `Arc`
    /// references, the backend's resources are released when the last reference
    /// is dropped (graceful drain via the backend's `Drop` impl).
    pub async fn remove(&mut self, model_id: &str) -> Result<(), String> {
        if self.backends.remove(model_id).is_some() {
            tracing::info!("Model '{}' removed from manager", model_id);
            if self.default_model.as_deref() == Some(model_id) {
                self.default_model = self.backends.keys().next().cloned();
            }
            Ok(())
        } else {
            Err(format!("Model '{}' not found", model_id))
        }
    }

    /// Get an `Arc` reference to a backend by model id, or the default.
    /// The caller can release the manager lock and keep using the returned
    /// `Arc` for inference without blocking other requests.
    pub fn get(&self, model_id: Option<&str>) -> Option<SharedBackend> {
        match model_id {
            Some(id) if !id.is_empty() => self.backends.get(id).cloned(),
            _ => self
                .default_model
                .as_ref()
                .and_then(|id| self.backends.get(id).cloned()),
        }
    }

    /// Get the default model id.
    pub fn default_id(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    /// Set the default model.
    pub fn set_default(&mut self, model_id: &str) -> Result<(), String> {
        if self.backends.contains_key(model_id) {
            self.default_model = Some(model_id.to_string());
            Ok(())
        } else {
            Err(format!("Model '{}' not found", model_id))
        }
    }

    /// List all loaded models.
    pub fn list(&self) -> Vec<LoadedModel> {
        self.backends
            .iter()
            .map(|(id, backend)| {
                let info = backend.model_info().unwrap_or(ModelInfo {
                    name: id.clone(),
                    context_size: 0,
                    gpu_layers: 0,
                    backend_name: backend.name().to_string(),
                });
                LoadedModel {
                    id: id.clone(),
                    model_info: info,
                    backend_name: backend.name().to_string(),
                }
            })
            .collect()
    }

    /// Check if any model is loaded.
    pub fn has_models(&self) -> bool {
        !self.backends.is_empty()
    }

    /// Get the number of loaded models.
    pub fn count(&self) -> usize {
        self.backends.len()
    }
}

/// Shared, thread-safe wrapper around ModelManager.
pub type SharedModelManager = Arc<Mutex<ModelManager>>;
