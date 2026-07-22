use athenas_core::{AppConfig, BackendType, HardwareDetector, ModelRegistry, Result};
use athenas_inference::{BackendFactory, ChatMessage, ChatRequest, ModelLoadConfig, StreamChunk};
use tokio::sync::mpsc;

pub async fn run(
    model: Option<String>,
    backend: BackendType,
    gpu_layers: i32,
    context_size: u32,
) -> Result<()> {
    let config = AppConfig::load()?;
    config.ensure_dirs()?;

    let hardware = HardwareDetector::detect()?;

    let model_path = if let Some(m) = model {
        resolve_model(&config, &m)?
    } else {
        print_model_selector(&config)?
    };

    let mut backend = BackendFactory::create(backend, &hardware)?;

    println!("Loading model: {}", model_path);
    let load_config = ModelLoadConfig {
        model_path: model_path.clone(),
        gpu_layers,
        context_size,
        batch_size: config.inference.default_batch_size,
        threads: config.inference.default_threads,
        flash_attention: config.inference.flash_attention,
        use_mmap: true,
        use_mlock: false,
        reasoning_enabled: config.inference.reasoning_enabled,
        reasoning_budget: config.inference.reasoning_budget,
    };

    backend.load_model(load_config).await?;
    println!("Model loaded! Backend: {}", backend.name());
    println!("Type your message. Use /quit to exit, /clear to reset.\n");

    let mut messages: Vec<ChatMessage> = Vec::new();

    let stdin = std::io::stdin();
    loop {
        print!(">>> ");
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut input = String::new();
        stdin
            .read_line(&mut input)
            .map_err(|e| athenas_core::AthenasError::InvalidInput(e.to_string()))?;
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            match input.as_str() {
                "/quit" | "/exit" => {
                    backend.unload_model().await?;
                    break;
                }
                "/clear" => {
                    messages.clear();
                    println!("\n--- Conversation cleared ---\n");
                    continue;
                }
                "/help" => {
                    println!("Commands: /quit, /clear, /help");
                    continue;
                }
                _ => {
                    println!("Unknown command: {}", input);
                    continue;
                }
            }
        }

        messages.push(ChatMessage::user(&input));

        let req = ChatRequest {
            model: String::new(),
            messages: messages.clone(),
            temperature: Some(config.inference.default_temperature),
            top_p: Some(config.inference.default_top_p),
            max_tokens: Some(config.inference.default_max_tokens),
            stream: true,
            stop: None,
            seed: None,
        };

        let (tx, mut rx) = mpsc::channel::<StreamChunk>(100);
        let backend_ref = &backend;
        let _ = backend_ref.chat_stream(req, tx).await;

        print!("AI: ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let mut full_response = String::new();
        let mut tps = 0.0f32;

        while let Some(chunk) = rx.recv().await {
            if chunk.done {
                if let Some(stats) = chunk.stats {
                    tps = stats.tokens_per_second;
                }
            } else {
                print!("{}", chunk.text);
                std::io::Write::flush(&mut std::io::stdout()).ok();
                full_response.push_str(&chunk.text);
            }
        }
        println!("\n[{:.1} tok/s]\n", tps);

        messages.push(ChatMessage::assistant(&full_response));
    }

    Ok(())
}

fn resolve_model(config: &AppConfig, model_id: &str) -> Result<String> {
    let registry = ModelRegistry::new(config.paths.models_dir.clone());

    // Try to find in local registry
    if let Ok(model) = registry.find_model(model_id) {
        return Ok(model.file_path.to_string_lossy().to_string());
    }

    // Check if it's a direct file path
    let path = std::path::Path::new(model_id);
    if path.exists() && path.is_file() {
        return Ok(model_id.to_string());
    }

    Err(athenas_core::AthenasError::ModelNotFound(
        model_id.to_string(),
    ))
}

fn print_model_selector(config: &AppConfig) -> Result<String> {
    let registry = ModelRegistry::new(config.paths.models_dir.clone());
    let models = registry.list_local_models()?;

    if models.is_empty() {
        eprintln!("No models found. Download a model first:");
        eprintln!("  athenas models pull <repo-id>");
        return Err(athenas_core::AthenasError::ModelNotFound(
            "No models available".to_string(),
        ));
    }

    println!("Available models:");
    for (i, model) in models.iter().enumerate() {
        println!("  [{}] {} ({})", i, model.name, model.format_size());
    }
    print!("\nSelect model number: ");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|e| athenas_core::AthenasError::InvalidInput(e.to_string()))?;
    let idx: usize = input
        .trim()
        .parse()
        .map_err(|_| athenas_core::AthenasError::InvalidInput("Invalid number".to_string()))?;

    models
        .get(idx)
        .map(|m| m.file_path.to_string_lossy().to_string())
        .ok_or_else(|| athenas_core::AthenasError::InvalidInput("Invalid selection".to_string()))
}
