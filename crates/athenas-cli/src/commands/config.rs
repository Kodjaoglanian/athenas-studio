use athenas_core::{AppConfig, Result};

pub async fn set(key: &str, value: &str) -> Result<()> {
    let mut config = AppConfig::load()?;

    match key {
        "inference.default_backend" => {
            config.inference.default_backend = match value {
                "llama.cpp" | "llamacpp" => athenas_core::BackendType::LlamaCpp,
                "vllm" => athenas_core::BackendType::Vllm,
                "onnx" => athenas_core::BackendType::Onnx,
                "auto" => athenas_core::BackendType::Auto,
                _ => {
                    return Err(athenas_core::AthenasError::InvalidInput(format!(
                        "Invalid backend: {}. Use llama.cpp, vllm, onnx, or auto",
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
        "check_updates" => {
            config.check_updates = match value {
                "true" | "1" => true,
                "false" | "0" => false,
                _ => {
                    return Err(athenas_core::AthenasError::InvalidInput(format!(
                        "Invalid boolean for check_updates: {}. Use true or false",
                        value
                    )))
                }
            };
        }
        _ => {
            return Err(athenas_core::AthenasError::InvalidInput(format!(
                "Unknown config key: {}",
                key
            )));
        }
    }

    config.save()?;
    if key == "huggingface.token" {
        let display = if value.is_empty() {
            "[cleared]"
        } else {
            "[set]"
        };
        println!("Set {} = {}", key, display);
    } else {
        println!("Set {} = {}", key, value);
    }
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

    // Track current [section] header so section.key lookups match the right table
    let mut current_section = String::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            current_section = trimmed[1..trimmed.len() - 1].trim().to_string();
            continue;
        }
        if let Some((k, _)) = trimmed.split_once(" = ") {
            let full_key = if current_section.is_empty() {
                k.trim().to_string()
            } else {
                format!("{}.{}", current_section, k.trim())
            };
            if full_key == key {
                println!("{}", trimmed);
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
