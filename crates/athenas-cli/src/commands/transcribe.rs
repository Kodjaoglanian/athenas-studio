use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

pub async fn run(
    audio_path: &str,
    model: &str,
    language: &str,
    format: &str,
    translate: bool,
) -> Result<()> {
    // Read audio file
    let audio_data = std::fs::read(audio_path)
        .with_context(|| format!("Failed to read audio file: {}", audio_path))?;

    let filename = std::path::Path::new(audio_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.wav")
        .to_string();

    // Resolve model path
    let model_path = resolve_model_path(model)?;

    println!("Transcribing: {}", filename);
    println!("Model: {}", model_path);
    println!("Language: {}", language);
    println!();

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner} {msg}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );
    spinner.set_message("Transcribing audio...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    // Create WhisperBackend (auto-downloads whisper-cli if needed)
    let backend = athenas_inference::WhisperBackend::new()
        .await
        .context("Failed to initialize whisper backend")?;

    let request = athenas_inference::TranscriptionRequest {
        audio_data,
        filename,
        model: model_path,
        language: if language == "auto" {
            None
        } else {
            Some(language.to_string())
        },
        response_format: Some(format.to_string()),
        translate: Some(translate),
        max_len: None,
    };

    let response = backend.transcribe(&request).await?;

    spinner.finish_with_message("Done");

    // Output based on format
    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&response)?;
            println!("{}", json);
        }
        "srt" => {
            for seg in &response.segments {
                println!("{}", seg.id + 1);
                println!(
                    "{} --> {}",
                    format_srt_time(seg.start),
                    format_srt_time(seg.end)
                );
                println!("{}", seg.text);
                println!();
            }
        }
        "vtt" => {
            println!("WEBVTT");
            println!();
            for seg in &response.segments {
                println!(
                    "{} --> {}",
                    format_vtt_time(seg.start),
                    format_vtt_time(seg.end)
                );
                println!("{}", seg.text);
                println!();
            }
        }
        _ => {
            // Plain text
            println!("{}", response.text);
        }
    }

    Ok(())
}

/// Resolve model name/ID to a file path.
/// If the input is a path that exists, use it directly.
/// Otherwise, search the model registry.
fn resolve_model_path(model: &str) -> Result<String> {
    // If it's a direct path that exists, use it
    if std::path::Path::new(model).exists() {
        return Ok(model.to_string());
    }

    // Search in model registry
    let config = athenas_core::config::AppConfig::load()?;
    let registry = athenas_core::model_registry::ModelRegistry::new(config.paths.models_dir);

    let model_info = registry
        .find_model(model)
        .with_context(|| format!("Model '{}' not found in registry", model))?;

    Ok(model_info.file_path.to_string_lossy().to_string())
}

/// Format seconds as SRT timestamp: HH:MM:SS,mmm
fn format_srt_time(secs: f64) -> String {
    let hours = (secs / 3600.0) as u64;
    let mins = ((secs % 3600.0) / 60.0) as u64;
    let seconds = (secs % 60.0) as u64;
    let millis = ((secs % 1.0) * 1000.0) as u64;
    format!("{:02}:{:02}:{:02},{:03}", hours, mins, seconds, millis)
}

/// Format seconds as VTT timestamp: HH:MM:SS.mmm
fn format_vtt_time(secs: f64) -> String {
    let hours = (secs / 3600.0) as u64;
    let mins = ((secs % 3600.0) / 60.0) as u64;
    let seconds = (secs % 60.0) as u64;
    let millis = ((secs % 1.0) * 1000.0) as u64;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, seconds, millis)
}
