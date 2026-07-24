use athenas_core::{AppConfig, BackendType, HardwareDetector, ModelRegistry, Result};
use athenas_inference::{BackendFactory, ModelLoadConfig};
use athenas_server::{ApiKeyManager, ApiServer, AuditLogger, ModelRouter, VectorStoreConfig};

#[allow(clippy::too_many_arguments)]
pub async fn run(
    model: String,
    host: &str,
    port: u16,
    backend_type: BackendType,
    gpu_layers: i32,
    context_size: u32,
    max_concurrent: Option<u32>,
    rate_limit: Option<u32>,
    timeout_secs: Option<u64>,
    max_body_size_mb: Option<u32>,
) -> Result<()> {
    let mut config = AppConfig::load()?;
    config.ensure_dirs()?;

    // Initialize OpenTelemetry tracing
    let otel_config = config.server.otel.clone();
    let _otel_provider = athenas_server::tracing_setup::init_tracing(&otel_config);

    if let Some(mc) = max_concurrent {
        config.server.max_concurrent_requests = mc;
    }
    if let Some(rl) = rate_limit {
        config.server.rate_limit_per_second = rl;
    }
    if let Some(t) = timeout_secs {
        config.server.request_timeout_secs = t;
    }
    if let Some(bs) = max_body_size_mb {
        config.server.max_body_size_mb = bs;
    }

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
        reasoning_enabled: config.inference.reasoning_enabled,
        reasoning_budget: config.inference.reasoning_budget,
        mmproj_path: None,
        lora_paths: Vec::new(),
        parallel_slots: 4,
    };

    backend.load_model(load_config).await?;
    println!("Model loaded with backend: {}", backend.name());

    let data_dir = std::path::PathBuf::from(&config.paths.data_dir);
    let api_key_mgr = ApiKeyManager::new(data_dir.clone());
    let model_router = ModelRouter::new();
    let audit_logger = AuditLogger::new(data_dir.clone(), 10000);

    // Check if vector store is enabled before moving config
    let vs_enabled = config.server.vector_store.enabled;
    let vs_max_docs = config.server.vector_store.max_documents;
    let vs_top_k = config.server.vector_store.default_top_k;

    print_startup_banner(&config, host, port, &hardware, &model);

    let mut server = ApiServer::new(config, backend)
        .with_api_key_manager(api_key_mgr)
        .with_model_router(model_router)
        .with_audit_logger(audit_logger);

    if vs_enabled {
        let vs_config = VectorStoreConfig {
            enabled: true,
            data_dir,
            max_documents: vs_max_docs,
            default_top_k: vs_top_k,
        };
        server = server.with_vector_store(vs_config);
    }

    server.start(host, port).await
}

fn print_startup_banner(
    config: &AppConfig,
    host: &str,
    port: u16,
    hardware: &athenas_core::HardwareInfo,
    model: &str,
) {
    println!();
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │                   Athenas Studio Server                 │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!(
        "  │  Endpoint:  http://{}:{}                        │",
        host, port
    );
    println!("  │  Model:     {:<44}│", truncate(model, 44));
    println!("  │  Backend:   {:<44}│", config.inference.default_backend);
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │  Hardware:                                              │");
    println!("  │    CPU threads: {:<38}│", hardware.cpus);
    if !hardware.gpus.is_empty() {
        let gpu_str = hardware
            .gpus
            .iter()
            .map(|g| g.name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        println!("  │    GPU:         {:<38}│", truncate(&gpu_str, 38));
    } else {
        println!("  │    GPU:         {:<38}│", "None (CPU-only)");
    }
    println!(
        "  │    Memory:      {:<38}│",
        format!("{} MB", hardware.memory_total_mb)
    );
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │  Server Config:                                         │");
    println!(
        "  │    Max concurrent: {:<34}│",
        config.server.max_concurrent_requests
    );
    println!(
        "  │    Rate limit:     {:<34}│",
        format!("{}/s", config.server.rate_limit_per_second)
    );
    println!(
        "  │    Timeout:        {:<34}│",
        format!("{}s", config.server.request_timeout_secs)
    );
    println!(
        "  │    Max body size:  {:<34}│",
        format!("{}MB", config.server.max_body_size_mb)
    );
    println!(
        "  │    Compression:    {:<34}│",
        if config.server.enable_compression {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "  │    Metrics:        {:<34}│",
        if config.server.enable_metrics {
            "enabled (/metrics)"
        } else {
            "disabled"
        }
    );
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │  Endpoints:                                             │");
    println!("  │    POST /v1/chat/completions   (OpenAI-compatible)      │");
    println!("  │    POST /v1/completions        (OpenAI-compatible)      │");
    println!("  │    POST /v1/embeddings         (OpenAI-compatible)      │");
    println!("  │    GET  /v1/models             (List loaded models)     │");
    println!("  │    GET  /v1/health             (Health + system info)   │");
    println!("  │    GET  /v1/ready              (Kubernetes readiness)   │");
    println!("  │    GET  /metrics               (Prometheus metrics)     │");
    println!("  │    POST /v1/keys               (Create API key)         │");
    println!("  │    GET  /v1/keys               (List API keys)          │");
    println!("  │    GET  /v1/audit/logs         (Query audit logs)       │");
    println!("  │    GET  /v1/audit/stats        (Audit statistics)       │");
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();
    println!("  Press Ctrl+C to stop.");
    println!();
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
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
