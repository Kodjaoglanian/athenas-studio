use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tracing::info;

use athenas_core::{AthenasError, Result};

use crate::client::HuggingFaceClient;

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub speed_mbps: f64,
    pub percent: Option<f64>,
}

pub struct ModelDownloader {
    client: HuggingFaceClient,
    models_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(client: HuggingFaceClient, models_dir: PathBuf) -> Self {
        Self { client, models_dir }
    }

    pub async fn download_model(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    ) -> Result<PathBuf> {
        let safe_repo = repo_id.replace('/', "__");
        let model_dir = self.models_dir.join(&safe_repo);
        std::fs::create_dir_all(&model_dir)?;

        let file_path = model_dir.join(filename);
        let download_url = self.client.download_url(repo_id, filename, revision);

        info!("Downloading {} from {}", filename, download_url);

        // Check if file already exists
        if file_path.exists() {
            info!("File already exists: {}", file_path.display());
            return Ok(file_path);
        }

        let client = self.client.client().clone();
        let mut req = client.get(&download_url);
        if let Some(token) = self.client.token() {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AthenasError::Download(format!("Download request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Download(format!(
                "Download failed: {} - {}",
                status, text
            )));
        }

        let total_bytes = resp.content_length();

        // Create temp file for download
        let temp_path = file_path.with_extension("part");
        let mut file = tokio::fs::File::create(&temp_path)
            .await
            .map_err(|e| AthenasError::Download(format!("Failed to create file: {}", e)))?;

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let start_time = std::time::Instant::now();
        let mut last_update = std::time::Instant::now();

        while let Some(chunk_result) = stream.next().await {
            let chunk =
                chunk_result.map_err(|e| AthenasError::Download(format!("Stream error: {}", e)))?;

            file.write_all(&chunk)
                .await
                .map_err(|e| AthenasError::Download(format!("Write error: {}", e)))?;

            downloaded += chunk.len() as u64;

            if let Some(ref tx) = progress_tx {
                let now = std::time::Instant::now();
                if now.duration_since(last_update) > std::time::Duration::from_millis(100) {
                    last_update = now;
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let speed_mbps = if elapsed > 0.0 {
                        (downloaded as f64 / (1024.0 * 1024.0)) / elapsed
                    } else {
                        0.0
                    };
                    let percent = total_bytes.map(|t| (downloaded as f64 / t as f64) * 100.0);

                    let _ = tx
                        .send(DownloadProgress {
                            downloaded_bytes: downloaded,
                            total_bytes,
                            speed_mbps,
                            percent,
                        })
                        .await;
                }
            }
        }

        file.flush()
            .await
            .map_err(|e| AthenasError::Download(format!("Flush error: {}", e)))?;
        drop(file);

        // Rename temp file to final
        tokio::fs::rename(&temp_path, &file_path)
            .await
            .map_err(|e| AthenasError::Download(format!("Rename error: {}", e)))?;

        // Send final progress
        if let Some(ref tx) = progress_tx {
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed_mbps = if elapsed > 0.0 {
                (downloaded as f64 / (1024.0 * 1024.0)) / elapsed
            } else {
                0.0
            };
            let _ = tx
                .send(DownloadProgress {
                    downloaded_bytes: downloaded,
                    total_bytes: Some(downloaded),
                    speed_mbps,
                    percent: Some(100.0),
                })
                .await;
        }

        info!("Downloaded {} ({} bytes)", filename, downloaded);
        Ok(file_path)
    }

    pub async fn list_gguf_files(
        &self,
        repo_id: &str,
        revision: &str,
    ) -> Result<Vec<(String, Option<u64>)>> {
        let files = self.client.get_model_files(repo_id, revision).await?;
        let gguf_files: Vec<(String, Option<u64>)> = files
            .iter()
            .filter(|f| f.r#type == "file" && f.path.ends_with(".gguf"))
            .map(|f| {
                let size = f.size.or(f.lfs.as_ref().and_then(|l| l.size));
                (f.path.clone(), size)
            })
            .collect();
        Ok(gguf_files)
    }

    pub async fn list_safetensors_files(
        &self,
        repo_id: &str,
        revision: &str,
    ) -> Result<Vec<(String, Option<u64>)>> {
        let files = self.client.get_model_files(repo_id, revision).await?;
        let st_files: Vec<(String, Option<u64>)> = files
            .iter()
            .filter(|f| {
                f.r#type == "file" && (f.path.ends_with(".safetensors") || f.path.ends_with(".bin"))
            })
            .map(|f| {
                let size = f.size.or(f.lfs.as_ref().and_then(|l| l.size));
                (f.path.clone(), size)
            })
            .collect();
        Ok(st_files)
    }
}
