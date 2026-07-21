use athenas_core::{AppConfig, Result};

pub async fn set(key: &str, value: &str) -> Result<()> {
    let mut config = AppConfig::load()?;

    match key {
        "inference.default_backend" => {
            config.inference.default_backend = match value {
                "llama.cpp" | "llamacpp" => athenas_core::BackendType::LlamaCpp,
                "vllm" => athenas_core::BackendType::Vllm,
                "auto" => athenas_core::BackendType::Auto,
                _ => {
                    return Err(athenas_core::AthenasError::InvalidInput(format!(
                        "Invalid backend: {}. Use llama.cpp, vllm, or auto",
                        value
                    )))
                }
            };
        }
        "inference.default_gpu_layers" => {
            config.inference.default_gpu_layers = value.parse().map_err(|_| {
                athenas_core::AthenasError::InvalidInput("Invalid number".to_string())
            })?;
        }
        "inference.default_context_size" => {
            config.inference.default_context_size = value.parse().map_err(|_| {
                athenas_core::AthenasError::InvalidInput("Invalid number".to_string())
            })?;
        }
        "inference.default_temperature" => {
            config.inference.default_temperature = value.parse().map_err(|_| {
                athenas_core::AthenasError::InvalidInput("Invalid float".to_string())
            })?;
        }
        "inference.default_max_tokens" => {
            config.inference.default_max_tokens = value.parse().map_err(|_| {
                athenas_core::AthenasError::InvalidInput("Invalid number".to_string())
            })?;
        }
        "inference.flash_attention" => {
            config.inference.flash_attention = value == "true" || value == "1";
        }
        "server.default_host" => {
            config.server.default_host = value.to_string();
        }
        "server.default_port" => {
            config.server.default_port = value.parse().map_err(|_| {
                athenas_core::AthenasError::InvalidInput("Invalid port".to_string())
            })?;
        }
        "server.api_key" => {
            config.server.api_key = if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            };
        }
        "huggingface.token" => {
            config.huggingface.token = if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            };
        }
        "logging.level" => {
            config.logging.level = value.to_string();
        }
        _ => {
            return Err(athenas_core::AthenasError::InvalidInput(format!(
                "Unknown config key: {}",
                key
            )));
        }
    }

    config.save()?;
    println!("Set {} = {}", key, value);
    Ok(())
}

pub async fn get(key: &str) -> Result<()> {
    let config = AppConfig::load()?;

    if key == "huggingface.token" {
        if config.huggingface.token.is_some() {
            println!("huggingface.token = [set]");
        } else if std::env::var("HF_TOKEN").is_ok() {
            println!("huggingface.token = [set via HF_TOKEN env]");
        } else {
            println!("huggingface.token = [not set]");
        }
        return Ok(());
    }

    let content = toml::to_string_pretty(&config)
        .map_err(|e| athenas_core::AthenasError::Config(e.to_string()))?;

    // Simple key lookup in the TOML output
    for line in content.lines() {
        if line.starts_with(&format!("{} = ", key)) {
            println!("{}", line);
            return Ok(());
        }
    }

    // Try nested
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() == 2 {
        for line in content.lines() {
            if line.starts_with(&format!("{} = ", parts[1])) {
                // Check if we're in the right section
                println!("{}", line);
                return Ok(());
            }
        }
    }

    println!("Key not found: {}", key);
    Ok(())
}

pub async fn show() -> Result<()> {
    let mut config = AppConfig::load()?;
    // Mask token for display
    if config.huggingface.token.is_some() {
        config.huggingface.token = Some("[set]".to_string());
    }
    let content = toml::to_string_pretty(&config)
        .map_err(|e| athenas_core::AthenasError::Config(e.to_string()))?;
    println!("{}", content);
    Ok(())
}

pub async fn init() -> Result<()> {
    let config = AppConfig::default();
    config.save()?;
    println!(
        "Configuration initialized at: {:?}",
        AppConfig::config_path()?
    );
    Ok(())
}

pub async fn login(token: Option<String>) -> Result<()> {
    let token = match token {
        Some(t) => t,
        None => {
            println!("Get your token at: https://huggingface.co/settings/tokens");
            print!("Enter your HuggingFace token: ");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(|e| athenas_core::AthenasError::InvalidInput(e.to_string()))?;
            input.trim().to_string()
        }
    };

    if token.is_empty() {
        println!("No token provided. Login cancelled.");
        return Ok(());
    }

    let mut config = AppConfig::load()?;
    config.huggingface.token = Some(token);
    config.save()?;

    println!("HuggingFace token saved to config.");
    println!("You can now access private and gated models.");
    Ok(())
}
