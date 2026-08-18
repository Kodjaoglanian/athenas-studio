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
    gpu_runtime: athenas_core::GpuRuntime,
    gpu_device: Option<u32>,
    context_size: u32,
    threads: Option<u32>,
    batch_size: Option<u32>,
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
    // Fall back to config values when CLI args are at defaults
    let effective_gpu_runtime = if gpu_runtime == athenas_core::GpuRuntime::Auto {
        config.inference.gpu_runtime
    } else {
        gpu_runtime
    };
    let effective_gpu_device = gpu_device.or(config.inference.gpu_device);

    // Apply auto resource limits if enabled
    let mut effective_threads = threads.unwrap_or(config.inference.default_threads);
    let mut effective_batch_size = batch_size.unwrap_or(config.inference.default_batch_size);
    // 0 = use config default (allows CLI to omit --context-size and still respect config.toml)
    let mut effective_context_size = if context_size == 0 {
        config.inference.default_context_size
    } else {
        context_size
    };

    // Resolve draft model path (relative to models_dir or absolute)
    let draft_model_path = config.inference.draft_model.as_ref().map(|p| {
        let path = std::path::Path::new(p);
        if path.is_absolute() {
            p.clone()
        } else {
            config
                .paths
                .models_dir
                .join(p)
                .to_string_lossy()
                .to_string()
        }
    });

    if config.inference.auto_resource_limits {
        // Cap threads: leave cpu_reserve_cores free for the OS
        if effective_threads == 0 {
            // 0 = auto: use all cores minus reserve
            effective_threads = hardware
                .cpus
                .saturating_sub(config.inference.cpu_reserve_cores)
                .max(1);
        } else {
            // User specified a value — cap it at (cpus - reserve)
            let max_threads = hardware
                .cpus
                .saturating_sub(config.inference.cpu_reserve_cores)
                .max(1);
            if effective_threads > max_threads {
                effective_threads = max_threads;
            }
        }

        // Cap context size based on available memory
        if hardware.memory_total_mb > 0 {
            let model_size_mb = std::fs::metadata(&model_path)
                .map(|m| m.len() / (1024 * 1024))
                .unwrap_or(0);
            let reserved = model_size_mb + config.inference.ram_reserve_mb;
            let usable = hardware.memory_total_mb.saturating_sub(reserved);
            // Rough: allow up to 50% of remaining RAM for context
            let max_ctx = ((usable * 1024) / (64 * 1024 / 1024)) as u32 * 1024;
            if max_ctx > 0 && effective_context_size > max_ctx {
                effective_context_size = max_ctx.max(512);
            }
        }

        // Cap batch size — can't exceed context size
        if effective_batch_size > effective_context_size {
            effective_batch_size = effective_context_size;
        }

        println!(
            "Auto resource limits: threads={}, context={}, batch={}",
            effective_threads, effective_context_size, effective_batch_size
        );
    }

    let load_config = ModelLoadConfig {
        model_path,
        gpu_layers,
        gpu_runtime: effective_gpu_runtime,
        gpu_device: effective_gpu_device,
        context_size: effective_context_size,
        batch_size: effective_batch_size,
        threads: effective_threads,
        flash_attention: config.inference.flash_attention,
        use_mmap: true,
        use_mlock: false,
        reasoning_enabled: config.inference.reasoning_enabled,
        reasoning_budget: config.inference.reasoning_budget,
        mmproj_path: None,
        lora_paths: config.inference.lora_paths.clone(),
        parallel_slots: config.inference.parallel_slots,
        draft_model_path,
        draft_max_tokens: config.inference.draft_max_tokens,
        draft_min_ctx: config.inference.draft_min_ctx,
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

    // Check if semantic cache is enabled
    let sc_enabled = config.server.semantic_cache.enabled;
    let sc_threshold = config.server.semantic_cache.similarity_threshold;
    let sc_ttl = config.server.semantic_cache.ttl_secs;
    let sc_max_entries = config.server.semantic_cache.max_entries;

    print_startup_banner(&config, host, port, &hardware, &model);

    let mut server = ApiServer::new(config, backend)
        .with_api_key_manager(api_key_mgr)
        .with_model_router(model_router)
        .with_audit_logger(audit_logger);

    if vs_enabled {
        let vs_config = VectorStoreConfig {
            enabled: true,
            data_dir: data_dir.clone(),
            max_documents: vs_max_docs,
            default_top_k: vs_top_k,
        };
        server = server.with_vector_store(vs_config);
    }

    if sc_enabled {
        let sc_config = athenas_server::SemanticCacheConfig {
            enabled: true,
            similarity_threshold: sc_threshold,
            ttl_secs: sc_ttl,
            max_entries: sc_max_entries,
            data_dir: data_dir.clone(),
        };
        let cache = athenas_server::SemanticCache::new(sc_config);
        server = server.with_semantic_cache(cache);
        tracing::info!(
            "Semantic cache enabled (threshold={:.2}, ttl={}s, max_entries={})",
            sc_threshold,
            sc_ttl,
            sc_max_entries
        );
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
