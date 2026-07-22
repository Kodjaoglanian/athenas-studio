use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, Semaphore};
use tracing::info;

use athenas_core::{AppConfig, Result};
use athenas_inference::Backend;

use crate::metrics::{Metrics, SharedMetrics};
use crate::middleware::{RateLimiter, SharedRateLimiter};
use crate::model_manager::{ModelManager, SharedModelManager};

pub struct ApiServer {
    config: AppConfig,
    model_manager: SharedModelManager,
    metrics: SharedMetrics,
    semaphore: Arc<Semaphore>,
    rate_limiter: SharedRateLimiter,
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

        Self {
            config,
            model_manager,
            metrics,
            semaphore,
            rate_limiter,
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

        Self {
            config,
            model_manager: manager,
            metrics,
            semaphore,
            rate_limiter,
        }
    }

    /// Get the shared model manager (for loading/unloading models at runtime).
    pub fn model_manager(&self) -> SharedModelManager {
        self.model_manager.clone()
    }

    pub async fn start(&self, host: &str, port: u16) -> Result<()> {
        let app = crate::routes::create_router(
            self.model_manager.clone(),
            self.metrics.clone(),
            self.semaphore.clone(),
            self.rate_limiter.clone(),
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
