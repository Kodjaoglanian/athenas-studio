use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tracing::info;

use athenas_core::{AthenasError, Result};

use crate::client::HuggingFaceClient;

const WRITE_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB write buffer
const PARALLEL_CHUNKS: u64 = 4; // Number of parallel download connections
const MIN_CHUNK_SIZE: u64 = 2 * 1024 * 1024; // 2 MB minimum chunk size

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

        // HEAD request to check if server supports range requests
        let supports_range = self.check_range_support(&download_url).await?;

        if supports_range {
            info!(
                "Server supports range requests, using parallel download ({} chunks)",
                PARALLEL_CHUNKS
            );
            self.parallel_download(&download_url, &file_path, progress_tx)
                .await
        } else {
            info!("Server does not support range requests, using single-stream download");
            self.single_stream_download(&download_url, &file_path, progress_tx)
                .await
        }
    }

    async fn check_range_support(&self, url: &str) -> Result<bool> {
        let client = self.client.client().clone();
        let mut req = client.head(url);
        if let Some(token) = self.client.token() {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AthenasError::Download(format!("HEAD request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Ok(false);
        }

        // Check Accept-Ranges header
        let accepts = resp
            .headers()
            .get("accept-ranges")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let has_length = resp.content_length().is_some();

        Ok(accepts.eq_ignore_ascii_case("bytes") && has_length)
    }

    async fn parallel_download(
        &self,
        url: &str,
        file_path: &PathBuf,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    ) -> Result<PathBuf> {
        // Get total file size from HEAD
        let client = self.client.client().clone();
        let mut head_req = client.head(url);
        if let Some(token) = self.client.token() {
            head_req = head_req.bearer_auth(token);
        }
        let head_resp = head_req
            .send()
            .await
            .map_err(|e| AthenasError::Download(format!("HEAD request failed: {}", e)))?;

        let total_bytes = head_resp
            .content_length()
            .ok_or_else(|| AthenasError::Download("No content-length in HEAD response".into()))?;

        // Calculate chunk boundaries
        let num_chunks = if total_bytes / PARALLEL_CHUNKS < MIN_CHUNK_SIZE {
            // File too small for parallel, use fewer chunks
            (total_bytes / MIN_CHUNK_SIZE).max(1)
        } else {
            PARALLEL_CHUNKS
        };

        let chunk_size = total_bytes / num_chunks;
        let last_chunk_size = total_bytes - chunk_size * (num_chunks - 1);

        info!(
            "File size: {} bytes, {} chunks of ~{} bytes each",
            total_bytes, num_chunks, chunk_size
        );

        // Pre-allocate the file
        let temp_path = file_path.with_extension("part");
        {
            let file = tokio::fs::File::create(&temp_path)
                .await
                .map_err(|e| AthenasError::Download(format!("Failed to create file: {}", e)))?;
            file.set_len(total_bytes)
                .await
                .map_err(|e| AthenasError::Download(format!("Failed to set file length: {}", e)))?;
        }

        // Shared progress counter
        let downloaded = Arc::new(AtomicU64::new(0));
        let start_time = std::time::Instant::now();

        // Spawn progress reporter task
        let progress_downloaded = downloaded.clone();
        let progress_handle = if let Some(ref tx) = progress_tx {
            let tx = tx.clone();
            let total = total_bytes;
            Some(tokio::spawn(async move {
                let mut last_update = std::time::Instant::now();
                let mut speed_window_bytes: u64 = 0;
                let mut speed_window_start = std::time::Instant::now();
                let mut last_downloaded: u64 = 0;

                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

                    let current = progress_downloaded.load(Ordering::Relaxed);
                    let now = std::time::Instant::now();

                    if now.duration_since(last_update) < std::time::Duration::from_millis(200) {
                        continue;
                    }
                    last_update = now;

                    let delta = current.saturating_sub(last_downloaded);
                    speed_window_bytes += delta;
                    last_downloaded = current;

                    let window_elapsed = speed_window_start.elapsed().as_secs_f64();
                    let speed_mbps = if window_elapsed > 0.0 {
                        (speed_window_bytes as f64 / (1024.0 * 1024.0)) / window_elapsed
                    } else {
                        0.0
                    };

                    if window_elapsed > 2.0 {
                        speed_window_bytes = 0;
                        speed_window_start = now;
                    }

                    let percent = (current as f64 / total as f64) * 100.0;

                    if tx
                        .send(DownloadProgress {
                            downloaded_bytes: current,
                            total_bytes: Some(total),
                            speed_mbps,
                            percent: Some(percent),
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }

                    if current >= total {
                        break;
                    }
                }
            }))
        } else {
            None
        };

        // Spawn download tasks for each chunk
        let mut tasks = Vec::new();
        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = if i == num_chunks - 1 {
                start + last_chunk_size
            } else {
                start + chunk_size
            };

            let url = url.to_string();
            let token = self.client.token().map(|t| t.to_string());
            let client = client.clone();
            let temp_path = temp_path.clone();
            let downloaded = downloaded.clone();

            tasks.push(tokio::spawn(async move {
                download_chunk(
                    &client,
                    &url,
                    token.as_deref(),
                    start,
                    end,
                    &temp_path,
                    downloaded,
                )
                .await
            }));
        }

        // Wait for all chunks
        let mut errors = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("Task join error: {}", e)),
            }
        }

        // Cancel progress reporter
        if let Some(handle) = progress_handle {
            handle.abort();
        }

        if !errors.is_empty() {
            let _ = tokio::fs::remove_file(&temp_path).await;
            return Err(AthenasError::Download(format!(
                "Parallel download failed: {}",
                errors.join("; ")
            )));
        }

        // Verify size
        let final_downloaded = downloaded.load(Ordering::Relaxed);
        if final_downloaded != total_bytes {
            let _ = tokio::fs::remove_file(&temp_path).await;
            return Err(AthenasError::Download(format!(
                "Download incomplete: {} / {} bytes",
                final_downloaded, total_bytes
            )));
        }

        // Rename temp file to final
        tokio::fs::rename(&temp_path, file_path)
            .await
            .map_err(|e| AthenasError::Download(format!("Rename error: {}", e)))?;

        // Send final progress
        if let Some(ref tx) = progress_tx {
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed_mbps = if elapsed > 0.0 {
                (total_bytes as f64 / (1024.0 * 1024.0)) / elapsed
            } else {
                0.0
            };
            let _ = tx
                .send(DownloadProgress {
                    downloaded_bytes: total_bytes,
                    total_bytes: Some(total_bytes),
                    speed_mbps,
                    percent: Some(100.0),
                })
                .await;
        }

        info!(
            "Downloaded {} ({} bytes) in {:.1}s",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            total_bytes,
            start_time.elapsed().as_secs_f64()
        );
        Ok(file_path.clone())
    }

    async fn single_stream_download(
        &self,
        url: &str,
        file_path: &PathBuf,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    ) -> Result<PathBuf> {
        let client = self.client.client().clone();
        let mut req = client.get(url);
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
        let temp_path = file_path.with_extension("part");
        let mut file = tokio::fs::File::create(&temp_path)
            .await
            .map_err(|e| AthenasError::Download(format!("Failed to create file: {}", e)))?;

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let start_time = std::time::Instant::now();
        let mut last_update = std::time::Instant::now();

        let mut write_buf: Vec<u8> = Vec::with_capacity(WRITE_BUFFER_SIZE);
        let mut speed_window_bytes: u64 = 0;
        let mut speed_window_start = std::time::Instant::now();

        while let Some(chunk_result) = stream.next().await {
            let chunk =
                chunk_result.map_err(|e| AthenasError::Download(format!("Stream error: {}", e)))?;

            write_buf.extend_from_slice(&chunk);
            downloaded += chunk.len() as u64;
            speed_window_bytes += chunk.len() as u64;

            if write_buf.len() >= WRITE_BUFFER_SIZE {
                file.write_all(&write_buf)
                    .await
                    .map_err(|e| AthenasError::Download(format!("Write error: {}", e)))?;
                write_buf.clear();
            }

            if let Some(ref tx) = progress_tx {
                let now = std::time::Instant::now();
                if now.duration_since(last_update) > std::time::Duration::from_millis(200) {
                    last_update = now;

                    let window_elapsed = speed_window_start.elapsed().as_secs_f64();
                    let speed_mbps = if window_elapsed > 0.0 {
                        (speed_window_bytes as f64 / (1024.0 * 1024.0)) / window_elapsed
                    } else {
                        0.0
                    };

                    if window_elapsed > 2.0 {
                        speed_window_bytes = 0;
                        speed_window_start = now;
                    }

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

        if !write_buf.is_empty() {
            file.write_all(&write_buf)
                .await
                .map_err(|e| AthenasError::Download(format!("Write error: {}", e)))?;
        }

        file.flush()
            .await
            .map_err(|e| AthenasError::Download(format!("Flush error: {}", e)))?;
        drop(file);

        tokio::fs::rename(&temp_path, file_path)
            .await
            .map_err(|e| AthenasError::Download(format!("Rename error: {}", e)))?;

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

        info!(
            "Downloaded {} ({} bytes)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            downloaded
        );
        Ok(file_path.clone())
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

/// Download a single chunk using HTTP Range request and write it to the correct file offset.
async fn download_chunk(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    start: u64,
    end: u64,
    file_path: &PathBuf,
    downloaded: Arc<AtomicU64>,
) -> Result<()> {
    let range = format!("bytes={}-{}", start, end - 1);
    let mut req = client.get(url).header("Range", &range);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| AthenasError::Download(format!("Chunk request failed: {}", e)))?;

    if !resp.status().is_success() && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(AthenasError::Download(format!(
            "Chunk download failed: {} - {}",
            status, text
        )));
    }

    use futures::StreamExt;
    let mut stream = resp.bytes_stream();

    // Open file and seek to the correct offset
    let file = tokio::fs::OpenOptions::new()
        .write(true)
        .open(file_path)
        .await
        .map_err(|e| AthenasError::Download(format!("Failed to open file for chunk: {}", e)))?;

    let file = Arc::new(Mutex::new(file));
    let mut writer = file.lock().await;
    use tokio::io::{AsyncSeekExt, AsyncWriteExt};
    writer
        .seek(std::io::SeekFrom::Start(start))
        .await
        .map_err(|e| AthenasError::Download(format!("Seek error: {}", e)))?;

    let mut write_buf: Vec<u8> = Vec::with_capacity(WRITE_BUFFER_SIZE);

    while let Some(chunk_result) = stream.next().await {
        let chunk =
            chunk_result.map_err(|e| AthenasError::Download(format!("Stream error: {}", e)))?;

        write_buf.extend_from_slice(&chunk);
        downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);

        if write_buf.len() >= WRITE_BUFFER_SIZE {
            writer
                .write_all(&write_buf)
                .await
                .map_err(|e| AthenasError::Download(format!("Write error: {}", e)))?;
            write_buf.clear();
        }
    }

    if !write_buf.is_empty() {
        writer
            .write_all(&write_buf)
            .await
            .map_err(|e| AthenasError::Download(format!("Write error: {}", e)))?;
    }

    writer
        .flush()
        .await
        .map_err(|e| AthenasError::Download(format!("Flush error: {}", e)))?;

    Ok(())
}
