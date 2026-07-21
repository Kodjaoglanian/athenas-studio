use athenas_core::{AppConfig, BackendType, HardwareDetector, ModelRegistry, Result};
use athenas_inference::{Backend, BackendFactory, ModelLoadConfig};
use athenas_server::ApiServer;

pub async fn run(
    model: String,
    host: &str,
    port: u16,
    backend_type: BackendType,
    gpu_layers: i32,
    context_size: u32,
) -> Result<()> {
    let config = AppConfig::load()?;
    config.ensure_dirs()?;

    let hardware = HardwareDetector::detect()?;

    let model_path = resolve_model(&config, &model)?;

    let mut backend = BackendFactory::create(backend_type, &hardware)?;

    println!("Loading model: {}", model_path);
    let load_config = ModelLoadConfig {
        model_path,
        gpu_layers,
        context_size,
        batch_size: config.inference.default_batch_size,
        threads: config.inference.default_threads,
        flash_attention: config.inference.flash_attention,
        use_mmap: true,
        use_mlock: false,
    };

    backend.load_model(load_config).await?;
    println!("Model loaded with backend: {}", backend.name());

    let server = ApiServer::new(config, backend);
    println!("\nStarting OpenAI-compatible API server on http://{}:{}", host, port);
    println!("Endpoints:");
    println!("  POST /v1/chat/completions");
    println!("  POST /v1/completions");
    println!("  GET  /v1/models");
    println!("  GET  /v1/health");
    println!("\nPress Ctrl+C to stop.\n");

    server.start(host, port).await
}

fn resolve_model(config: &AppConfig, model_id: &str) -> Result<String> {
    let registry = ModelRegistry::new(config.paths.models_dir.clone());
    if let Ok(model) = registry.find_model(model_id) {
        return Ok(model.file_path.to_string_lossy().to_string());
    }
    let path = std::path::Path::new(model_id);
    if path.exists() && path.is_file() {
        return Ok(model_id.to_string());
    }
    Err(athenas_core::AthenasError::ModelNotFound(model_id.to_string()))
}
