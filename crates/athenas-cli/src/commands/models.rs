use athenas_core::{AppConfig, ModelRegistry, Result};
use comfy_table::{presets::UTF8_FULL, Table};
use indicatif::{ProgressBar, ProgressStyle};

use athenas_hub::{HuggingFaceClient, ModelDownloader, ModelSearchFilters};

pub async fn list() -> Result<()> {
    let config = AppConfig::load()?;
    let registry = ModelRegistry::new(config.paths.models_dir.clone());
    let models = registry.list_local_models()?;

    if models.is_empty() {
        println!("No models downloaded.");
        println!("Use 'athenas models search <query>' to find models on HuggingFace.");
        println!("Then use 'athenas models pull <repo-id>' to download.");
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Name", "Repo", "Format", "Size", "Quantization"]);

    for model in &models {
        table.add_row(vec![
            model.name.clone(),
            model.repo_id.clone(),
            model.format.to_string(),
            model.format_size(),
            model.quantization.clone().unwrap_or("-".to_string()),
        ]);
    }

    println!("{}", table);

    let disk = registry.disk_usage()?;
    println!(
        "\nTotal disk usage: {:.2} GB",
        disk as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    Ok(())
}

pub async fn search(query: &str, pipeline: Option<String>, gguf: bool) -> Result<()> {
    let config = AppConfig::load()?;
    let client = HuggingFaceClient::new(config.huggingface.token.clone());

    let filters = ModelSearchFilters {
        pipeline_tag: pipeline,
        library_name: None,
        gguf_only: gguf,
        safetensors_only: false,
    };

    println!("Searching HuggingFace for '{}'...\n", query);
    let results = client.search_models(query, &filters).await?;

    if results.is_empty() {
        println!("No models found.");
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Model ID", "Downloads", "Likes", "Pipeline", "Tags"]);

    for result in results.iter().take(30) {
        let tags = result
            .tags
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        table.add_row(vec![
            result.id.clone(),
            format_downloads(result.downloads),
            result.likes.to_string(),
            result.pipeline_tag.clone(),
            tags,
        ]);
    }

    println!("{}", table);
    println!("\nTo download: athenas models pull <repo-id>");

    Ok(())
}

pub async fn pull(repo_id: &str, file: Option<String>, revision: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let client = HuggingFaceClient::new(config.huggingface.token.clone());
    let downloader = ModelDownloader::new(client.clone(), config.paths.models_dir.clone());

    // Determine which file to download
    let filename = if let Some(f) = file {
        f
    } else {
        // Try to find GGUF files
        let gguf_files = downloader.list_gguf_files(repo_id, revision).await?;

        if gguf_files.is_empty() {
            // Try safetensors
            let st_files = downloader.list_safetensors_files(repo_id, revision).await?;
            if st_files.is_empty() {
                return Err(athenas_core::AthenasError::Download(format!(
                    "No model files found in {}",
                    repo_id
                )));
            }
            // Pick the largest safetensors file
            st_files
                .iter()
                .max_by_key(|(_, size)| size.unwrap_or(0))
                .map(|(name, _)| name.clone())
                .unwrap_or_default()
        } else if gguf_files.len() == 1 {
            gguf_files[0].0.clone()
        } else {
            // Multiple GGUF files — let user pick
            println!("Multiple GGUF files found in {}:", repo_id);
            for (i, (name, size)) in gguf_files.iter().enumerate() {
                let size_str = size
                    .map(|s| format!("{:.2} GB", s as f64 / 1e9))
                    .unwrap_or("?".to_string());
                println!("  [{}] {} ({})", i, name, size_str);
            }
            print!("\nSelect file number: ");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(|e| athenas_core::AthenasError::InvalidInput(e.to_string()))?;
            let idx: usize = input.trim().parse().map_err(|_| {
                athenas_core::AthenasError::InvalidInput("Invalid number".to_string())
            })?;
            gguf_files
                .get(idx)
                .map(|(name, _)| name.clone())
                .ok_or_else(|| {
                    athenas_core::AthenasError::InvalidInput("Invalid selection".to_string())
                })?
        }
    };

    // Get file size for progress bar
    let files = client.get_model_files(repo_id, revision).await?;
    let file_info = files.iter().find(|f| f.path == filename);
    let total_size = file_info.and_then(|f| f.size.or(f.lfs.as_ref().and_then(|l| l.size)));

    let pb = if let Some(size) = total_size {
        ProgressBar::new(size)
    } else {
        ProgressBar::new_spinner()
    };

    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}) {msg}"
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let (tx, mut rx) = tokio::sync::mpsc::channel::<athenas_hub::DownloadProgress>(10);

    let pb_clone = pb.clone();
    let progress_task = tokio::spawn(async move {
        while let Some(progress) = rx.recv().await {
            pb_clone.set_position(progress.downloaded_bytes);
            if let Some(speed) = Some(progress.speed_mbps) {
                pb_clone.set_message(format!("{:.1} MB/s", speed));
            }
        }
        pb_clone.finish_with_message("Download complete");
    });

    println!("Downloading {} from {}/{}", filename, repo_id, revision);
    let path = downloader
        .download_model(repo_id, &filename, revision, Some(tx))
        .await?;

    progress_task.await.ok();

    println!("\nModel saved to: {}", path.display());
    println!("You can now use it with: athenas chat {}", filename);

    Ok(())
}

pub async fn remove(model: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let registry = ModelRegistry::new(config.paths.models_dir.clone());

    let model_info = registry.find_model(model)?;
    println!("Removing: {}", model_info.name);
    println!("  Path: {}", model_info.file_path.display());
    println!("  Size: {}", model_info.format_size());

    print!("Are you sure? (y/N): ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();
    if input.trim().eq_ignore_ascii_case("y") {
        registry.remove_model(model)?;
        println!("Model removed.");
    } else {
        println!("Cancelled.");
    }

    Ok(())
}

pub async fn info(model: &str) -> Result<()> {
    let config = AppConfig::load()?;
    let registry = ModelRegistry::new(config.paths.models_dir.clone());
    let model_info = registry.find_model(model)?;
    println!("{}", model_info);
    Ok(())
}

fn format_downloads(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
