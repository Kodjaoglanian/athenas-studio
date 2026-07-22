use athenas_core::{AppConfig, BackendType, HardwareDetector, ModelRegistry, Result};
use athenas_inference::{BackendFactory, ChatMessage, ChatRequest, ModelLoadConfig};
use comfy_table::{presets::UTF8_FULL, Table};
use std::time::Instant;

pub async fn list() -> Result<()> {
    let hw = HardwareDetector::detect()?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Backend", "Available", "Details"]);

    // llama.cpp
    let llamacpp_available = hw.has_cuda || hw.has_rocm || hw.has_vulkan || hw.has_metal;
    let llamacpp_details = if hw.has_cuda {
        "CUDA (NVIDIA GPU)"
    } else if hw.has_rocm {
        "ROCm (AMD GPU)"
    } else if hw.has_vulkan {
        "Vulkan"
    } else if hw.has_metal {
        "Metal (Apple)"
    } else {
        "CPU only"
    };
    table.add_row(vec![
        "llama.cpp",
        if llamacpp_available {
            "✓"
        } else {
            "✓ (CPU)"
        },
        llamacpp_details,
    ]);

    // vLLM
    let vllm_available = hw.has_cuda || hw.has_rocm;
    let vllm_details = if hw.has_cuda {
        "CUDA (NVIDIA GPU)"
    } else if hw.has_rocm {
        "ROCm (AMD GPU)"
    } else {
        "Requires CUDA or ROCm"
    };
    table.add_row(vec![
        "vLLM",
        if vllm_available { "✓" } else { "✗" },
        vllm_details,
    ]);

    println!("{}", table);

    if !hw.gpus.is_empty() {
        println!("\nGPUs:");
        for gpu in &hw.gpus {
            println!("  {}", gpu);
        }
    }

    Ok(())
}

const BENCHMARK_PROMPTS: &[&str] = &[
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "What is the capital of France?",
    "Describe the process of photosynthesis.",
    "List three benefits of exercise.",
];

pub async fn benchmark(model: Option<String>) -> Result<()> {
    let hw = HardwareDetector::detect()?;

    println!("\n  Athenas Studio — Backend Benchmark\n  ====================================\n");

    println!("  Hardware:");
    println!("    CPUs:          {}", hw.cpus);
    println!(
        "    Memory:        {}MB / {}MB",
        hw.memory_available_mb, hw.memory_total_mb
    );
    println!(
        "    CUDA:          {}",
        if hw.has_cuda { "yes" } else { "no" }
    );
    println!(
        "    ROCm:          {}",
        if hw.has_rocm { "yes" } else { "no" }
    );
    println!(
        "    Vulkan:        {}",
        if hw.has_vulkan { "yes" } else { "no" }
    );
    println!(
        "    Metal:         {}",
        if hw.has_metal { "yes" } else { "no" }
    );

    if !hw.gpus.is_empty() {
        println!("    GPUs:");
        for gpu in &hw.gpus {
            println!("      - {}", gpu);
        }
    }

    let model_path = match model {
        Some(m) => resolve_model(&AppConfig::load()?, &m)?,
        None => {
            println!("\n  No model specified. Use --model <path> to benchmark.");
            println!("  Example: athenas backend benchmark --model llama-2-7b-chat.Q4_K_M.gguf\n");
            return Ok(());
        }
    };

    let config = AppConfig::load()?;

    println!("\n  Model: {}", model_path);
    println!("  Backend: auto (resolves to llama.cpp)\n");

    let mut backend = BackendFactory::create(BackendType::Auto, &hw)?;

    let load_start = Instant::now();
    backend
        .load_model(ModelLoadConfig {
            model_path: model_path.clone(),
            gpu_layers: config.inference.default_gpu_layers,
            context_size: config.inference.default_context_size,
            batch_size: config.inference.default_batch_size,
            threads: config.inference.default_threads,
            flash_attention: config.inference.flash_attention,
            use_mmap: true,
            use_mlock: false,
            reasoning_enabled: config.inference.reasoning_enabled,
            reasoning_budget: config.inference.reasoning_budget,
        })
        .await?;
    let load_time = load_start.elapsed();

    println!("  Model loaded in {:.2}s", load_time.as_secs_f64());
    println!("  Backend: {}\n", backend.name());

    if let Some(info) = backend.model_info() {
        println!("  Model info:");
        println!("    Context size:  {}", info.context_size);
        println!("    GPU layers:    {}", info.gpu_layers);
        println!();
    }

    println!(
        "  Running {} benchmark prompts...\n",
        BENCHMARK_PROMPTS.len()
    );

    let mut results: Vec<BenchResult> = Vec::new();

    for (i, prompt) in BENCHMARK_PROMPTS.iter().enumerate() {
        print!(
            "  [{}/{}] \"{}\"... ",
            i + 1,
            BENCHMARK_PROMPTS.len(),
            prompt
        );
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let req = ChatRequest {
            model: String::new(),
            messages: vec![ChatMessage::user(*prompt)],
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(128),
            stream: false,
            stop: None,
            seed: Some(42),
        };

        let start = Instant::now();
        let response = backend.chat(req).await?;
        let elapsed = start.elapsed();

        let result = BenchResult {
            prompt_idx: i,
            prompt_tokens: response.stats.tokens_prompt,
            generated_tokens: response.stats.tokens_generated,
            total_time_ms: elapsed.as_millis() as u64,
            tps: response.stats.tokens_per_second,
        };

        println!(
            "{:.1} tok/s ({} tokens in {}ms)",
            result.tps, result.generated_tokens, result.total_time_ms
        );

        results.push(result);
    }

    backend.unload_model().await?;

    // Summary
    let avg_tps: f32 = results.iter().map(|r| r.tps).sum::<f32>() / results.len() as f32;
    let avg_latency: f64 =
        results.iter().map(|r| r.total_time_ms as f64).sum::<f64>() / results.len() as f64;
    let total_tokens: u32 = results.iter().map(|r| r.generated_tokens).sum();

    println!("\n  ────────────────────────────────────────");
    println!("  Benchmark Summary");
    println!("  ────────────────────────────────────────");
    println!("  Backend:           {}", backend.name());
    println!("  Model load time:   {:.2}s", load_time.as_secs_f64());
    println!("  Prompts run:       {}", results.len());
    println!("  Total tokens:      {}", total_tokens);
    println!("  Avg tokens/sec:    {:.2}", avg_tps);
    println!("  Avg latency:       {:.0}ms", avg_latency);
    println!("  ────────────────────────────────────────\n");

    // Detailed table
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        "#",
        "Prompt tokens",
        "Generated",
        "Time (ms)",
        "Tok/s",
    ]);

    for r in &results {
        table.add_row(vec![
            format!("{}", r.prompt_idx + 1),
            r.prompt_tokens.to_string(),
            r.generated_tokens.to_string(),
            r.total_time_ms.to_string(),
            format!("{:.2}", r.tps),
        ]);
    }

    println!("{}", table);

    Ok(())
}

struct BenchResult {
    prompt_idx: usize,
    prompt_tokens: u32,
    generated_tokens: u32,
    total_time_ms: u64,
    tps: f32,
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
