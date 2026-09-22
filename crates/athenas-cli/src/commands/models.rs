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

    // Recursive listing — ONNX models live in variant subdirectories.
    let all_files = client.get_model_files_recursive(repo_id, revision).await?;
    let onnx_variant_dirs: Vec<String> = {
        let mut dirs: Vec<String> = all_files
            .iter()
            .filter(|f| f.r#type == "file" && f.path.ends_with(".onnx"))
            .map(|f| {
                f.path
                    .rsplit_once('/')
                    .map(|(d, _)| d.to_string())
                    .unwrap_or_default()
            })
            .collect();
        dirs.sort();
        dirs.dedup();
        dirs
    };
    let has_gguf = all_files
        .iter()
        .any(|f| f.r#type == "file" && f.path.ends_with(".gguf"));

    // Resolve whether this pull is an ONNX directory download: --file naming
    // a variant dir, or a repo with ONNX files and no GGUF to pick.
    let onnx_variant: Option<String> = match file.as_deref().map(|f| f.trim_matches('/')) {
        Some(f)
            if !f.ends_with(".onnx")
                && !f.ends_with(".gguf")
                && !f.ends_with(".safetensors")
                && !f.ends_with(".bin") =>
        {
            if onnx_variant_dirs.iter().any(|d| d == f) {
                Some(f.to_string())
            } else {
                return Err(athenas_core::AthenasError::InvalidInput(format!(
                    "'{}' is not a model file or ONNX variant dir in {}",
                    f, repo_id
                )));
            }
        }
        Some(_) => None,
        None if onnx_variant_dirs.is_empty() || has_gguf => None,
        None if onnx_variant_dirs.len() == 1 => Some(onnx_variant_dirs[0].clone()),
        None => {
            println!("Multiple ONNX variants found in {}:", repo_id);
            for (i, d) in onnx_variant_dirs.iter().enumerate() {
                let label = if d.is_empty() { "(root)" } else { d.as_str() };
                let size: u64 = all_files
                    .iter()
                    .filter(|f| {
                        f.r#type == "file"
                            && f.path
                                .rsplit_once('/')
                                .map(|(dir, _)| dir == d)
                                .unwrap_or_else(|| d.is_empty())
                    })
                    .map(|f| f.size.or(f.lfs.as_ref().and_then(|l| l.size)).unwrap_or(0))
                    .sum();
                println!("  [{}] {} ({:.2} GB)", i, label, size as f64 / 1e9);
            }
            print!("\nSelect variant number: ");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .map_err(|e| athenas_core::AthenasError::InvalidInput(e.to_string()))?;
            let idx: usize = input.trim().parse().map_err(|_| {
                athenas_core::AthenasError::InvalidInput("Invalid number".to_string())
            })?;
            Some(onnx_variant_dirs.get(idx).cloned().ok_or_else(|| {
                athenas_core::AthenasError::InvalidInput("Invalid selection".to_string())
            })?)
        }
    };

    if let Some(variant) = onnx_variant {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bytes} ({bytes_per_sec}) {msg}",
            )
            .unwrap(),
        );
        let (tx, mut rx) = tokio::sync::mpsc::channel::<athenas_hub::DownloadProgress>(10);
        let pb_clone = pb.clone();
        let progress_task = tokio::spawn(async move {
            while let Some(progress) = rx.recv().await {
                pb_clone.set_position(progress.downloaded_bytes);
                pb_clone.set_message(format!("{:.1} MB/s", progress.speed_mbps));
            }
            pb_clone.finish_with_message("Download complete");
        });

        println!(
            "Downloading ONNX variant '{}' from {}/{}",
            variant, repo_id, revision
        );
        let path = downloader
            .download_model_dir(repo_id, revision, &variant, Some(tx))
            .await?;
        progress_task.await.ok();
        println!("\nModel saved to: {}", path.display());
        println!("You can now use it with: athenas chat {}", repo_id);
        return Ok(());
    }

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

    // Get file size for progress bar and sha256 for integrity check
    let file_info = all_files.iter().find(|f| f.path == filename);
    let total_size = file_info.and_then(|f| f.size.or(f.lfs.as_ref().and_then(|l| l.size)));
    let expected_sha = file_info.and_then(|f| f.lfs.as_ref().and_then(|l| l.sha256.clone()));

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
        .download_model_verify(repo_id, &filename, revision, Some(tx), expected_sha)
        .await?;

    progress_task.await.ok();

    println!("\nModel saved to: {}", path.display());

    // Auto-download mmproj file if present in the repo (for multimodal models)
    let mmproj_files: Vec<_> = all_files
        .iter()
        .filter(|f| {
            f.r#type == "file"
                && f.path.to_lowercase().contains("mmproj")
                && (f.path.ends_with(".gguf") || f.path.ends_with(".bin"))
        })
        .collect();

    if !mmproj_files.is_empty() {
        for mmproj in &mmproj_files {
            // Check if already downloaded
            let safe_repo = repo_id.replace('/', "__");
            let mmproj_dir = config.paths.models_dir.join(&safe_repo);
            let mmproj_local = mmproj_dir.join(&mmproj.path);

            if mmproj_local.exists() {
                println!("mmproj already present: {}", mmproj.path);
                continue;
            }

            println!("\nDownloading multimodal projector: {}", mmproj.path);

            let (tx2, mut rx2) = tokio::sync::mpsc::channel::<athenas_hub::DownloadProgress>(10);
            let pb2 = if let Some(size) = mmproj.size.or(mmproj.lfs.as_ref().and_then(|l| l.size)) {
                ProgressBar::new(size)
            } else {
                ProgressBar::new_spinner()
            };
            pb2.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}) {msg}",
                )
                .unwrap()
                .progress_chars("=>-"),
            );
            let pb2_clone = pb2.clone();
            let progress_task2 = tokio::spawn(async move {
                while let Some(progress) = rx2.recv().await {
                    pb2_clone.set_position(progress.downloaded_bytes);
                    if let Some(speed) = Some(progress.speed_mbps) {
                        pb2_clone.set_message(format!("{:.1} MB/s", speed));
                    }
                }
                pb2_clone.finish_with_message("mmproj download complete");
            });

            if let Err(e) = downloader
                .download_model_verify(
                    repo_id,
                    &mmproj.path,
                    revision,
                    Some(tx2),
                    mmproj.lfs.as_ref().and_then(|l| l.sha256.clone()),
                )
                .await
            {
                println!(
                    "\nWarning: failed to download mmproj ({}): {}",
                    mmproj.path, e
                );
            } else {
                println!("\nmmproj saved alongside model — vision support enabled.");
            }
            progress_task2.await.ok();
        }
    }

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
