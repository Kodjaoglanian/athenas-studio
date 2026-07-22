use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use athenas_core::{AthenasError, Result};

use crate::client::HuggingFaceClient;

const WRITE_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB write buffer
const PARALLEL_CHUNKS: u64 = 8; // Number of parallel download connections
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
            match self
                .parallel_download(&download_url, &file_path, progress_tx.clone())
                .await
            {
                Ok(path) => return Ok(path),
                Err(e) => {
                    warn!(
                        "Parallel download failed ({}), falling back to single-stream",
                        e
                    );
                    // Clean up any partial files
                    let temp_path = file_path.with_extension("part");
                    let _ = tokio::fs::remove_file(&temp_path).await;
                }
            }
        }

        info!("Using single-stream download");
        self.single_stream_download(&download_url, &file_path, progress_tx)
            .await
    }

    async fn check_range_support(&self, url: &str) -> Result<bool> {
        let client = self.client.client().clone();
        let mut req = client.get(url).header("Range", "bytes=0-0");
        if let Some(token) = self.client.token() {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AthenasError::Download(format!("Range probe failed: {}", e)))?;

        let status = resp.status();
        debug!("Range probe status: {}", status);
        debug!("Response headers: {:?}", resp.headers());

        // 206 Partial Content means server supports Range requests
        if status == reqwest::StatusCode::PARTIAL_CONTENT {
            // Extract total size from Content-Range header: "bytes 0-0/12345"
            if let Some(cr) = resp.headers().get("content-range") {
                if let Ok(s) = cr.to_str() {
                    debug!("Content-Range: {}", s);
                }
            }
            return Ok(true);
        }

        // Some servers return 200 and ignore the Range header
        warn!(
            "Server does not support Range requests (status: {}), falling back to single-stream",
            status
        );
        Ok(false)
    }

    async fn parallel_download(
        &self,
        url: &str,
        file_path: &PathBuf,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    ) -> Result<PathBuf> {
        let client = self.client.client().clone();
        let token = self.client.token().map(|t| t.to_string());

        // Get total file size via a Range probe (bytes=0-0)
        // Content-Range header format: "bytes 0-0/12345678"
        let mut probe_req = client.get(url).header("Range", "bytes=0-0");
        if let Some(ref t) = token {
            probe_req = probe_req.bearer_auth(t);
        }
        let probe_resp = probe_req
            .send()
            .await
            .map_err(|e| AthenasError::Download(format!("Range probe failed: {}", e)))?;

        // Extract the final URL after redirects (HuggingFace LFS redirects to CDN)
        let cdn_url = probe_resp.url().to_string();
        if cdn_url != url {
            info!("Resolved CDN URL: {}", cdn_url);
        }

        let total_bytes = probe_resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split('/').nth(1))
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| AthenasError::Download("No Content-Range in probe response".into()))?;

        info!(
            "File size: {} bytes ({:.1} MB)",
            total_bytes,
            total_bytes as f64 / (1024.0 * 1024.0)
        );

        // Calculate chunk boundaries
        let num_chunks = if total_bytes / PARALLEL_CHUNKS < MIN_CHUNK_SIZE {
            (total_bytes / MIN_CHUNK_SIZE).max(1)
        } else {
            PARALLEL_CHUNKS
        };

        let chunk_size = total_bytes / num_chunks;
        let last_chunk_size = total_bytes - chunk_size * (num_chunks - 1);

        info!(
            "Using {} parallel chunks of ~{:.1} MB each",
            num_chunks,
            chunk_size as f64 / (1024.0 * 1024.0)
        );

        // Shared progress counter
        let downloaded = Arc::new(AtomicU64::new(0));
        let start_time = std::time::Instant::now();

        // Spawn progress reporter task
        let progress_downloaded = downloaded.clone();
        let progress_handle = if let Some(ref tx) = progress_tx {
            let tx = tx.clone();
            let total = total_bytes;
            Some(tokio::spawn(async move {
                let mut speed_window_bytes: u64 = 0;
                let mut speed_window_start = std::time::Instant::now();
                let mut last_downloaded: u64 = 0;

                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

                    let current = progress_downloaded.load(Ordering::Relaxed);
                    let now = std::time::Instant::now();

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

        // Each chunk downloads to its own temp file — no Mutex needed
        let temp_dir = file_path.parent().unwrap_or(file_path).to_path_buf();
        let mut tasks = Vec::new();
        let mut chunk_paths = Vec::new();

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let end = if i == num_chunks - 1 {
                start + last_chunk_size
            } else {
                start + chunk_size
            };

            let chunk_path = temp_dir.join(format!(".athenas_chunk_{}", i));
            chunk_paths.push(chunk_path.clone());

            let cdn_url = cdn_url.clone();
            let client = client.clone();
            let downloaded = downloaded.clone();
            let token = token.clone();

            tasks.push(tokio::spawn(async move {
                download_chunk_to_file(
                    &client,
                    &cdn_url,
                    token.as_deref(),
                    start,
                    end,
                    &chunk_path,
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
            for p in &chunk_paths {
                let _ = tokio::fs::remove_file(p).await;
            }
            return Err(AthenasError::Download(format!(
                "Parallel download failed: {}",
                errors.join("; ")
            )));
        }

        // Verify size
        let final_downloaded = downloaded.load(Ordering::Relaxed);
        if final_downloaded != total_bytes {
            for p in &chunk_paths {
                let _ = tokio::fs::remove_file(p).await;
            }
            return Err(AthenasError::Download(format!(
                "Download incomplete: {} / {} bytes",
                final_downloaded, total_bytes
            )));
        }

        // Concatenate chunk files into final file
        info!("Concatenating {} chunk files...", num_chunks);
        let temp_path = file_path.with_extension("part");
        {
            let mut out = tokio::fs::File::create(&temp_path).await.map_err(|e| {
                AthenasError::Download(format!("Failed to create output file: {}", e))
            })?;

            for p in &chunk_paths {
                let mut chunk_file = tokio::fs::File::open(p).await.map_err(|e| {
                    AthenasError::Download(format!("Failed to open chunk file: {}", e))
                })?;
                tokio::io::copy(&mut chunk_file, &mut out)
                    .await
                    .map_err(|e| AthenasError::Download(format!("Concat error: {}", e)))?;
            }
            out.flush()
                .await
                .map_err(|e| AthenasError::Download(format!("Flush error: {}", e)))?;
        }

        // Cleanup chunk files
        for p in &chunk_paths {
            let _ = tokio::fs::remove_file(p).await;
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
            "Downloaded {} ({} bytes) in {:.1}s ({:.1} MB/s avg)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            total_bytes,
            start_time.elapsed().as_secs_f64(),
            (total_bytes as f64 / (1024.0 * 1024.0)) / start_time.elapsed().as_secs_f64()
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

/// Download a single chunk using HTTP Range request and write it to a dedicated temp file.
/// Each chunk gets its own file — no Mutex or seeking needed, enabling true parallel I/O.
async fn download_chunk_to_file(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    start: u64,
    end: u64,
    chunk_path: &PathBuf,
    downloaded: Arc<AtomicU64>,
) -> Result<()> {
    let chunk_start_time = std::time::Instant::now();
    let range = format!("bytes={}-{}", start, end - 1);
    let mut req = client.get(url).header("Range", &range);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| AthenasError::Download(format!("Chunk request failed: {}", e)))?;

    if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(AthenasError::Download(format!(
            "Chunk download failed: {} - {}",
            status, text
        )));
    }

    use futures::StreamExt;
    let mut stream = resp.bytes_stream();

    // Write to our own dedicated file — no lock contention
    let mut writer = tokio::fs::File::create(chunk_path)
        .await
        .map_err(|e| AthenasError::Download(format!("Failed to create chunk file: {}", e)))?;

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

    debug!(
        "Chunk bytes={}..{} ({} MB) in {:.1}s ({:.1} MB/s)",
        start,
        end,
        (end - start) / (1024 * 1024),
        chunk_start_time.elapsed().as_secs_f64(),
        (end - start) as f64 / (1024.0 * 1024.0) / chunk_start_time.elapsed().as_secs_f64()
    );
    Ok(())
}
