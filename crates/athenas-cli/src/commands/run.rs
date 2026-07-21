use athenas_core::{AppConfig, BackendType, HardwareDetector, ModelRegistry, Result};
use athenas_inference::{BackendFactory, ChatMessage, ChatRequest, ModelLoadConfig};

pub async fn run(
    model: String,
    prompt: &str,
    backend_type: BackendType,
    temperature: f32,
    max_tokens: u32,
    gpu_layers: i32,
) -> Result<()> {
    let config = AppConfig::load()?;
    config.ensure_dirs()?;

    let hardware = HardwareDetector::detect()?;

    let model_path = resolve_model(&config, &model)?;

    let mut backend = BackendFactory::create(backend_type, &hardware)?;

    let load_config = ModelLoadConfig {
        model_path,
        gpu_layers,
        context_size: config.inference.default_context_size,
        batch_size: config.inference.default_batch_size,
        threads: config.inference.default_threads,
        flash_attention: config.inference.flash_attention,
        use_mmap: true,
        use_mlock: false,
    };

    backend.load_model(load_config).await?;

    let req = ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage::user(prompt)],
        temperature: Some(temperature),
        top_p: Some(config.inference.default_top_p),
        max_tokens: Some(max_tokens),
        stream: false,
        stop: None,
        seed: None,
    };

    let response = backend.chat(req).await?;

    println!("{}", response.message.content);
    eprintln!(
        "\n--- {} tokens in {:.1} tok/s ---",
        response.stats.tokens_generated, response.stats.tokens_per_second
    );

    backend.unload_model().await?;
    Ok(())
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
    Err(athenas_core::AthenasError::ModelNotFound(
        model_id.to_string(),
    ))
}
