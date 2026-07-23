use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, Semaphore};
use tracing::info;

use athenas_core::{AppConfig, Result};
use athenas_inference::Backend;

use crate::metrics::{Metrics, SharedMetrics};
use crate::middleware::{RateLimiter, SharedRateLimiter};
use crate::model_manager::{ModelManager, SharedModelManager};
use crate::session_manager::{SessionManager, SharedSessionManager};
use crate::slot_manager::SlotManager;

pub struct ApiServer {
    config: AppConfig,
    model_manager: SharedModelManager,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    rate_limiter: SharedRateLimiter,
    session_manager: SharedSessionManager,
    slot_manager: Option<Arc<SlotManager>>,
}

impl ApiServer {
    pub fn new(config: AppConfig, backend: Box<dyn Backend>) -> Self {
        let metrics = Arc::new(Metrics::new());
        let semaphore = Arc::new(Semaphore::new(
            config.server.max_concurrent_requests as usize,
        ));
        let rate_limiter = Arc::new(RateLimiter::new(
            config.server.rate_limit_per_second,
            config.server.rate_limit_per_second,
        ));

        let model_manager = Arc::new(Mutex::new(ModelManager::with_default(backend)));
        let session_manager = Arc::new(Mutex::new(SessionManager::new(100, 4)));

        Self {
            config,
            model_manager,
            metrics,
            semaphore,
            rate_limiter,
            session_manager,
            slot_manager: None,
        }
    }

    /// Create an ApiServer with an existing shared model manager.
    pub fn with_manager(config: AppConfig, manager: SharedModelManager) -> Self {
        let metrics = Arc::new(Metrics::new());
        let semaphore = Arc::new(Semaphore::new(
            config.server.max_concurrent_requests as usize,
        ));
        let rate_limiter = Arc::new(RateLimiter::new(
            config.server.rate_limit_per_second,
            config.server.rate_limit_per_second,
        ));

        let session_manager = Arc::new(Mutex::new(SessionManager::new(100, 4)));

        Self {
            config,
            model_manager: manager,
            metrics,
            semaphore,
            rate_limiter,
            session_manager,
            slot_manager: None,
        }
    }

    /// Attach a slot manager (for KV cache persistence with llama-server).
    pub fn with_slot_manager(mut self, slot_mgr: SlotManager) -> Self {
        self.slot_manager = Some(Arc::new(slot_mgr));
        self
    }

    /// Get the shared model manager (for loading/unloading models at runtime).
    pub fn model_manager(&self) -> SharedModelManager {
        self.model_manager.clone()
    }

    /// Get the shared session manager.
    pub fn session_manager(&self) -> SharedSessionManager {
        self.session_manager.clone()
    }

    /// Get the slot manager (if any).
    pub fn slot_manager(&self) -> Option<Arc<SlotManager>> {
        self.slot_manager.clone()
    }

    pub async fn start(&self, host: &str, port: u16) -> Result<()> {
        let app = crate::routes::create_router(
            self.model_manager.clone(),
            self.metrics.clone(),
            self.semaphore.clone(),
            self.rate_limiter.clone(),
            self.session_manager.clone(),
            self.slot_manager.clone(),
            &self.config.server,
        );

        let addr = format!("{}:{}", host, port);
        info!("Starting API server on http://{}", addr);

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| athenas_core::AthenasError::Server(format!("Failed to bind: {}", e)))?;

        axum::serve(listener, app)
            .await
            .map_err(|e| athenas_core::AthenasError::Server(format!("Server error: {}", e)))?;

        Ok(())
    }
}

#[allow(dead_code)]
fn _unused_timeout(secs: u64) -> Duration {
    Duration::from_secs(secs)
}
