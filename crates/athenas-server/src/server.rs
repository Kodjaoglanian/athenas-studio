use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

use athenas_core::{AppConfig, Result};
use athenas_inference::Backend;

pub struct ApiServer {
    config: AppConfig,
    backend: Arc<Mutex<Box<dyn Backend>>>,
}

impl ApiServer {
    pub fn new(config: AppConfig, backend: Box<dyn Backend>) -> Self {
        Self {
            config,
            backend: Arc::new(Mutex::new(backend)),
        }
    }

    pub async fn start(&self, host: &str, port: u16) -> Result<()> {
        let app = crate::routes::create_router(
            self.backend.clone(),
            self.config.server.cors_enabled,
            self.config.server.api_key.clone(),
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
