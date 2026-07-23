mod commands;

use clap::{Parser, Subcommand};
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "athenas",
    about = "Athenas Studio — CLI/TUI for LLM inference (CUDA/ROCm/vLLM)",
    version,
    long_about = "A powerful CLI/TUI tool for running LLM models locally with CUDA, ROCm, and vLLM support. Compatible with HuggingFace model hub and OpenAI API."
)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Enable debug logging
    #[arg(short, long, global = true)]
    debug: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the TUI (default action)
    Tui,

    /// Start an interactive chat session
    Chat {
        /// Model file path or model ID
        model: Option<String>,
        /// Backend to use
        #[arg(long, value_enum, default_value = "auto")]
        backend: athenas_core::BackendType,
        /// GPU layers to offload (-1 for all)
        #[arg(long, default_value = "-1")]
        gpu_layers: i32,
        /// Context size
        #[arg(long, default_value = "4096")]
        context_size: u32,
    },

    /// Start the OpenAI-compatible API server
    Serve {
        /// Model file path or model ID
        model: String,
        /// Host to bind
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Backend to use
        #[arg(long, value_enum, default_value = "auto")]
        backend: athenas_core::BackendType,
        /// GPU layers to offload
        #[arg(long, default_value = "-1")]
        gpu_layers: i32,
        /// Context size
        #[arg(long, default_value = "4096")]
        context_size: u32,
        /// Max concurrent requests (semaphore)
        #[arg(long)]
        max_concurrent: Option<u32>,
        /// Rate limit (requests per second per IP)
        #[arg(long)]
        rate_limit: Option<u32>,
        /// Request timeout in seconds
        #[arg(long)]
        timeout: Option<u64>,
        /// Max request body size in MB
        #[arg(long)]
        max_body_size: Option<u32>,
    },

    /// Run a one-shot inference
    Run {
        /// Model file path or model ID
        model: String,
        /// Prompt text
        prompt: String,
        /// Backend to use
        #[arg(long, value_enum, default_value = "auto")]
        backend: athenas_core::BackendType,
        /// Temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Max tokens to generate
        #[arg(long, default_value = "2048")]
        max_tokens: u32,
        /// GPU layers
        #[arg(long, default_value = "-1")]
        gpu_layers: i32,
    },

    /// Manage models
    Models {
        #[command(subcommand)]
        action: ModelsCommands,
    },

    /// Configure backends and settings
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },

    /// Show hardware information
    Hardware,

    /// List available backends
    Backend {
        #[command(subcommand)]
        action: Option<BackendCommands>,
    },

    /// Login to HuggingFace Hub (set access token)
    Login {
        /// HuggingFace access token (skip prompt if provided)
        #[arg(long)]
        token: Option<String>,
    },

    /// Update athenas to the latest release
    Update,
}

#[derive(Subcommand)]
enum ModelsCommands {
    /// List locally downloaded models
    List,

    /// Search for models on HuggingFace
    Search {
        /// Search query
        query: String,
        /// Filter by pipeline tag (e.g., text-generation)
        #[arg(long)]
        pipeline: Option<String>,
        /// Filter to GGUF models only
        #[arg(long)]
        gguf: bool,
    },

    /// Download a model from HuggingFace
    Pull {
        /// HuggingFace repo ID (e.g., TheBloke/Llama-2-7B-Chat-GGUF)
        repo_id: String,
        /// Specific file to download (for GGUF, e.g., llama-2-7b-chat.Q4_K_M.gguf)
        #[arg(long)]
        file: Option<String>,
        /// Revision/branch
        #[arg(long, default_value = "main")]
        revision: String,
    },

    /// Remove a local model
    Remove {
        /// Model ID or name
        model: String,
    },

    /// Show detailed info about a model
    Info {
        /// Model ID or name
        model: String,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Set a configuration value
    Set { key: String, value: String },
    /// Get a configuration value
    Get { key: String },
    /// Show full configuration
    Show,
    /// Initialize/reset configuration
    Init,
}

#[derive(Subcommand)]
enum BackendCommands {
    /// List available backends
    List,
    /// Benchmark backends
    Benchmark {
        /// Model to use for benchmarking
        #[arg(long)]
        model: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let command = cli.command.unwrap_or(Commands::Tui);

    let is_tui = matches!(command, Commands::Tui);

    let filter = if cli.debug {
        EnvFilter::new("debug")
    } else if cli.verbose {
        EnvFilter::new("info")
    } else {
        EnvFilter::new("warn")
    };

    // For TUI mode, use a custom subscriber that captures logs into a buffer
    // instead of writing to stderr (which would corrupt the terminal display).
    // For other modes, use the standard fmt subscriber.
    if is_tui {
        let log_buffer = athenas_tui::log_buffer::LogBuffer::new(500);
        let buffer_layer = athenas_tui::log_buffer::LogBufferLayer::new(log_buffer.clone());

        tracing_subscriber::registry()
            .with(EnvFilter::new(if cli.debug {
                "debug"
            } else if cli.verbose {
                "info"
            } else {
                "warn"
            }))
            .with(buffer_layer)
            .init();

        commands::tui::run_with_log_buffer(log_buffer).await?;
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    if !is_tui {
        match command {
            Commands::Chat {
                model,
                backend,
                gpu_layers,
                context_size,
            } => commands::chat::run(model, backend, gpu_layers, context_size).await?,
            Commands::Serve {
                model,
                host,
                port,
                backend,
                gpu_layers,
                context_size,
                max_concurrent,
                rate_limit,
                timeout,
                max_body_size,
            } => {
                commands::serve::run(
                    model,
                    &host,
                    port,
                    backend,
                    gpu_layers,
                    context_size,
                    max_concurrent,
                    rate_limit,
                    timeout,
                    max_body_size,
                )
                .await?
            }
            Commands::Run {
                model,
                prompt,
                backend,
                temperature,
                max_tokens,
                gpu_layers,
            } => {
                commands::run::run(model, &prompt, backend, temperature, max_tokens, gpu_layers)
                    .await?
            }
            Commands::Models { action } => match action {
                ModelsCommands::List => commands::models::list().await?,
                ModelsCommands::Search {
                    query,
                    pipeline,
                    gguf,
                } => commands::models::search(&query, pipeline, gguf).await?,
                ModelsCommands::Pull {
                    repo_id,
                    file,
                    revision,
                } => commands::models::pull(&repo_id, file, &revision).await?,
                ModelsCommands::Remove { model } => commands::models::remove(&model).await?,
                ModelsCommands::Info { model } => commands::models::info(&model).await?,
            },
            Commands::Config { action } => match action {
                ConfigCommands::Set { key, value } => commands::config::set(&key, &value).await?,
                ConfigCommands::Get { key } => commands::config::get(&key).await?,
                ConfigCommands::Show => commands::config::show().await?,
                ConfigCommands::Init => commands::config::init().await?,
            },
            Commands::Hardware => commands::hardware::show().await?,
            Commands::Backend { action } => match action {
                Some(BackendCommands::List) => commands::backend::list().await?,
                Some(BackendCommands::Benchmark { model }) => {
                    commands::backend::benchmark(model).await?
                }
                None => commands::backend::list().await?,
            },
            Commands::Login { token } => commands::config::login(token).await?,
            Commands::Update => commands::update::run().await?,
            Commands::Tui => {} // already handled above
        }
    }

    Ok(())
}
