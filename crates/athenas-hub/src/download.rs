use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use athenas_core::{AthenasError, Result};

use crate::client::HuggingFaceClient;

const WRITE_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB write buffer

/// Join a repo-relative remote path onto a local dir, rejecting anything
/// that would escape it (`..`, absolute paths, Windows separators/drives).
fn safe_join(base: &std::path::Path, rel: &str) -> Result<PathBuf> {
    if rel.is_empty() || rel.starts_with('/') || rel.starts_with('\\') || rel.contains('\\') {
        return Err(AthenasError::Download(format!("unsafe path: {rel}")));
    }
    let mut out = base.to_path_buf();
    for comp in std::path::Path::new(rel).components() {
        match comp {
            std::path::Component::Normal(c) => out.push(c),
            _ => return Err(AthenasError::Download(format!("unsafe path: {rel}"))),
        }
    }
    Ok(out)
}
const PARALLEL_CHUNKS: u64 = 8; // Number of parallel download connections
const MIN_CHUNK_SIZE: u64 = 2 * 1024 * 1024; // 2 MB minimum chunk size
const MAX_RETRIES: usize = 3; // Max retries per chunk on stream error
const RETRY_BASE_DELAY_MS: u64 = 500; // Base delay for exponential backoff

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
        self.download_model_verify(repo_id, filename, revision, progress_tx, None)
            .await
    }

    /// Download a model file and verify its SHA256 against the expected
    /// digest from the HF API (lfs.sha256). On mismatch the file is
    /// removed and an error returned.
    pub async fn download_model_verify(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
        expected_sha256: Option<String>,
    ) -> Result<PathBuf> {
        let safe_repo = repo_id.replace('/', "__");
        let model_dir = self.models_dir.join(&safe_repo);
        std::fs::create_dir_all(&model_dir)?;

        // Filenames may carry a subdir (e.g. `onnx/model.onnx`) — keep the
        // structure but never allow escaping the model dir.
        let file_path = safe_join(&model_dir, filename)?;
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.download_to_path(
            repo_id,
            filename,
            revision,
            &file_path,
            progress_tx,
            expected_sha256,
        )
        .await
    }

    /// Download a single remote file to an absolute destination path,
    /// verifying SHA256 when provided.
    async fn download_to_path(
        &self,
        repo_id: &str,
        filename: &str,
        revision: &str,
        file_path: &PathBuf,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
        expected_sha256: Option<String>,
    ) -> Result<PathBuf> {
        let download_url = self.client.download_url(repo_id, filename, revision);

        info!("Downloading {} from {}", filename, download_url);

        // Check if file already exists
        if file_path.exists() {
            info!("File already exists: {}", file_path.display());
            self.verify_sha256(file_path, expected_sha256.as_deref())
                .await?;
            return Ok(file_path.clone());
        }

        // HEAD request to check if server supports range requests
        let supports_range = self.check_range_support(&download_url).await?;

        if supports_range {
            info!(
                "Server supports range requests, using parallel download ({} chunks)",
                PARALLEL_CHUNKS
            );
            match self
                .parallel_download(&download_url, file_path, progress_tx.clone())
                .await
            {
                Ok(path) => {
                    self.verify_sha256(&path, expected_sha256.as_deref())
                        .await?;
                    return Ok(path);
                }
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
        let path = self
            .single_stream_download(&download_url, file_path, progress_tx)
            .await?;
        self.verify_sha256(&path, expected_sha256.as_deref())
            .await?;
        Ok(path)
    }

    /// Download all files of an ONNX model *variant* — a model may be a
    /// directory with the `.onnx` + external `.onnx_data` weights +
    /// tokenizer + configs. `variant_dir` is the repo-relative subdir holding
    /// the variant ("" when the files are at the repo root).
    ///
    /// Files land in `models/<repo>/<variant-name>/` (variant name = last
    /// path component) or directly in `models/<repo>/` when `variant_dir` is
    /// empty. Root-level shared files (tokenizer, configs) are copied into
    /// the variant dir too so it's self-contained for ORT.
    ///
    /// Returns the local model directory.
    pub async fn download_model_dir(
        &self,
        repo_id: &str,
        revision: &str,
        variant_dir: &str,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    ) -> Result<PathBuf> {
        let files = self
            .client
            .get_model_files_recursive(repo_id, revision)
            .await?;

        let variant_dir = variant_dir.trim_matches('/');
        let prefix = if variant_dir.is_empty() {
            String::new()
        } else {
            format!("{}/", variant_dir)
        };

        // Files belonging to the variant: everything under its subdir, or
        // root-level files when the variant is the repo root.
        let variant_files: Vec<_> = files
            .iter()
            .filter(|f| {
                if f.r#type != "file" {
                    return false;
                }
                if prefix.is_empty() {
                    !f.path.contains('/')
                } else {
                    f.path.starts_with(&prefix)
                }
            })
            .collect();
        if variant_files.is_empty() {
            return Err(AthenasError::Download(format!(
                "no files found for variant '{}' in {}",
                if variant_dir.is_empty() {
                    "<root>"
                } else {
                    variant_dir
                },
                repo_id
            )));
        }

        // Root-level shared files every variant needs (tokenizer, configs).
        const SUPPORT: &[&str] = &[
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "genai_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "preprocessor_config.json",
            "processor_config.json",
            "vocab.txt",
        ];
        let root_support: Vec<_> = files
            .iter()
            .filter(|f| {
                f.r#type == "file" && !f.path.contains('/') && SUPPORT.contains(&f.path.as_str())
            })
            .filter(|f| prefix.is_empty() || !variant_files.iter().any(|v| v.path == f.path))
            .collect();

        let safe_repo = repo_id.replace('/', "__");
        let dest_dir = if variant_dir.is_empty() {
            self.models_dir.join(&safe_repo)
        } else {
            let variant_name = variant_dir.rsplit('/').next().unwrap_or(variant_dir);
            self.models_dir.join(&safe_repo).join(variant_name)
        };
        std::fs::create_dir_all(&dest_dir)?;

        // (remote_path, local_relative_path, sha256, size)
        let mut plan: Vec<(String, String, Option<String>, u64)> = Vec::new();
        for f in &variant_files {
            let rel = f.path.strip_prefix(&prefix).unwrap_or(&f.path).to_string();
            let size = f.lfs.as_ref().and_then(|l| l.size).or(f.size).unwrap_or(0);
            let sha = f.lfs.as_ref().and_then(|l| l.sha256.clone());
            plan.push((f.path.clone(), rel, sha, size));
        }
        for f in &root_support {
            let size = f.lfs.as_ref().and_then(|l| l.size).or(f.size).unwrap_or(0);
            let sha = f.lfs.as_ref().and_then(|l| l.sha256.clone());
            plan.push((f.path.clone(), f.path.clone(), sha, size));
        }

        let total_bytes: u64 = plan.iter().map(|(_, _, _, s)| *s).sum();
        info!(
            "Downloading ONNX model dir {} ({}) — {} files, {:.1} MB total",
            repo_id,
            if variant_dir.is_empty() {
                "<root>"
            } else {
                variant_dir
            },
            plan.len(),
            total_bytes as f64 / (1024.0 * 1024.0)
        );

        let mut completed_bytes: u64 = 0;
        for (remote_path, rel, sha, size) in &plan {
            let dest = safe_join(&dest_dir, rel)?;
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // Forward per-file progress translated into cumulative progress.
            let offset = completed_bytes;
            let (inner_tx, mut inner_rx) = tokio::sync::mpsc::channel::<DownloadProgress>(32);
            let fwd = progress_tx.clone().map(|tx| {
                tokio::spawn(async move {
                    while let Some(mut p) = inner_rx.recv().await {
                        p.downloaded_bytes += offset;
                        p.total_bytes = Some(total_bytes);
                        p.percent =
                            Some((p.downloaded_bytes as f64 / total_bytes.max(1) as f64) * 100.0);
                        if tx.send(p).await.is_err() {
                            break;
                        }
                    }
                })
            });

            let r = self
                .download_to_path(
                    repo_id,
                    remote_path,
                    revision,
                    &dest,
                    Some(inner_tx),
                    sha.clone(),
                )
                .await;
            if let Some(h) = fwd {
                h.abort();
            }
            r?;
            completed_bytes += size;
        }

        if let Some(ref tx) = progress_tx {
            let _ = tx
                .send(DownloadProgress {
                    downloaded_bytes: total_bytes,
                    total_bytes: Some(total_bytes),
                    speed_mbps: 0.0,
                    percent: Some(100.0),
                })
                .await;
        }

        Ok(dest_dir)
    }

    /// Verify a downloaded file against an expected SHA256 hex digest.
    /// Runs in spawn_blocking since model files can be many GB.
    /// On mismatch the file is deleted.
    async fn verify_sha256(&self, path: &PathBuf, expected: Option<&str>) -> Result<()> {
        let Some(expected) = expected else {
            return Ok(());
        };
        let expected = expected.to_lowercase();
        let hash_path = path.clone();
        let actual = tokio::task::spawn_blocking(move || {
            use sha2::Digest;
            let mut file = std::fs::File::open(&hash_path)?;
            let mut hasher = sha2::Sha256::new();
            std::io::copy(&mut file, &mut hasher)?;
            Ok::<String, std::io::Error>(hex::encode(hasher.finalize()))
        })
        .await
        .map_err(|e| AthenasError::Download(format!("SHA256 task failed: {}", e)))?
        .map_err(|e| AthenasError::Download(format!("SHA256 read failed: {}", e)))?;

        if actual == expected {
            info!("SHA256 verified: {}", &actual[..16.min(actual.len())]);
            return Ok(());
        }

        warn!(
            "SHA256 mismatch for {}: expected {}, got {} — removing file",
            path.display(),
            expected,
            actual
        );
        let _ = tokio::fs::remove_file(&path).await;
        Err(AthenasError::Download(format!(
            "SHA256 mismatch: expected {}, got {}",
            expected, actual
        )))
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

        // Note: We intentionally do NOT use the CDN URL from the redirect for chunk
        // downloads. HuggingFace LFS CDN signed URLs reject arbitrary range requests
        // with 403 "invalid range". Instead, each chunk request goes to the original
        // URL and gets a fresh redirect with a signed URL for that specific range.

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

            // Resume support: if chunk file exists with correct size, skip download
            let expected_size = end - start;
            if let Ok(meta) = std::fs::metadata(&chunk_path) {
                if meta.len() == expected_size {
                    info!(
                        "Chunk {} already complete ({} bytes), skipping",
                        i, expected_size
                    );
                    downloaded.fetch_add(expected_size, Ordering::Relaxed);
                    continue;
                }
            }

            let chunk_url = url.to_string();
            let client = client.clone();
            let downloaded = downloaded.clone();
            let token = token.clone();

            tasks.push(tokio::spawn(async move {
                download_chunk_with_retry(
                    &client,
                    &chunk_url,
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
            // Keep completed chunk files for resume on next attempt.
            // Only delete incomplete ones.
            for (i, p) in chunk_paths.iter().enumerate() {
                let expected = if i as u64 == num_chunks - 1 {
                    last_chunk_size
                } else {
                    chunk_size
                };
                let actual = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                if actual != expected {
                    let _ = tokio::fs::remove_file(p).await;
                }
            }
            return Err(AthenasError::Download(format!(
                "Parallel download failed (re-run to resume): {}",
                errors.join("; ")
            )));
        }

        // Verify size
        let final_downloaded = downloaded.load(Ordering::Relaxed);
        if final_downloaded != total_bytes {
            // Keep completed chunks for resume
            for (i, p) in chunk_paths.iter().enumerate() {
                let expected = if i as u64 == num_chunks - 1 {
                    last_chunk_size
                } else {
                    chunk_size
                };
                let actual = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                if actual != expected {
                    let _ = tokio::fs::remove_file(p).await;
                }
            }
            return Err(AthenasError::Download(format!(
                "Download incomplete: {} / {} bytes (re-run to resume)",
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
        let temp_path = file_path.with_extension("part");

        // Resume support: check if .part file exists and get its size
        let existing_bytes = match tokio::fs::metadata(&temp_path).await {
            Ok(meta) => meta.len(),
            Err(_) => 0,
        };

        let mut req = client.get(url);
        if let Some(token) = self.client.token() {
            req = req.bearer_auth(token);
        }
        // If we have a partial file, request the remaining bytes
        if existing_bytes > 0 {
            info!(
                "Resuming download from byte {} ({:.1} MB already downloaded)",
                existing_bytes,
                existing_bytes as f64 / (1024.0 * 1024.0)
            );
            req = req.header("Range", format!("bytes={}-", existing_bytes));
        }
        let resp = req
            .send()
            .await
            .map_err(|e| AthenasError::Download(format!("Download request failed: {}", e)))?;

        // Accept both 200 (full content) and 206 (partial content for resume)
        let is_resume = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Download(format!(
                "Download failed: {} - {}",
                status, text
            )));
        }

        // If server ignored Range and sent 200, start from scratch
        let (initial_bytes, mut file) = if is_resume {
            let file = tokio::fs::OpenOptions::new()
                .append(true)
                .open(&temp_path)
                .await
                .map_err(|e| {
                    AthenasError::Download(format!("Failed to open partial file: {}", e))
                })?;
            (existing_bytes, file)
        } else {
            let file = tokio::fs::File::create(&temp_path)
                .await
                .map_err(|e| AthenasError::Download(format!("Failed to create file: {}", e)))?;
            (0u64, file)
        };

        let total_bytes = if is_resume {
            // Content-Range: bytes start-end/total
            resp.headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.split('/').nth(1))
                .and_then(|s| s.parse::<u64>().ok())
        } else {
            resp.content_length()
        };

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = initial_bytes;
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

/// Download a chunk with automatic retry on failure.
/// On each retry, resumes from the partial chunk file if it exists.
async fn download_chunk_with_retry(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    start: u64,
    end: u64,
    chunk_path: &PathBuf,
    downloaded: Arc<AtomicU64>,
) -> Result<()> {
    let expected_size = end - start;

    for attempt in 0..=MAX_RETRIES {
        // Check if chunk is already complete from a previous run
        if let Ok(meta) = std::fs::metadata(chunk_path) {
            if meta.len() == expected_size {
                info!("Chunk {}-{} already complete, skipping", start, end);
                return Ok(());
            }
        }

        // Calculate resume offset within this chunk
        let chunk_have = std::fs::metadata(chunk_path).map(|m| m.len()).unwrap_or(0);
        let resume_from = start + chunk_have;
        let range = if chunk_have > 0 && resume_from < end {
            info!(
                "Retrying chunk {}-{} from byte {} (attempt {}/{})",
                start,
                end,
                resume_from,
                attempt + 1,
                MAX_RETRIES + 1
            );
            format!("bytes={}-{}", resume_from, end - 1)
        } else {
            if attempt > 0 {
                info!(
                    "Retrying chunk {}-{} from scratch (attempt {}/{})",
                    start,
                    end,
                    attempt + 1,
                    MAX_RETRIES + 1
                );
            }
            format!("bytes={}-{}", start, end - 1)
        };

        let mut req = client.get(url).header("Range", &range);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Chunk {}-{} attempt {} request failed: {}",
                    start,
                    end,
                    attempt + 1,
                    e
                );
                if attempt < MAX_RETRIES {
                    let delay = RETRY_BASE_DELAY_MS * (1 << attempt);
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    continue;
                }
                return Err(AthenasError::Download(format!(
                    "Chunk request failed after {} retries: {}",
                    MAX_RETRIES, e
                )));
            }
        };

        if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            warn!(
                "Chunk {}-{} attempt {} bad status: {} {}",
                start,
                end,
                attempt + 1,
                status,
                text
            );
            if attempt < MAX_RETRIES {
                let delay = RETRY_BASE_DELAY_MS * (1 << attempt);
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                // Delete partial file to restart clean
                let _ = tokio::fs::remove_file(chunk_path).await;
                continue;
            }
            return Err(AthenasError::Download(format!(
                "Chunk download failed after {} retries: {} - {}",
                MAX_RETRIES, status, text
            )));
        }

        // Open file for append if resuming, otherwise create fresh
        let mut writer = if chunk_have > 0 && resume_from < end {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(chunk_path)
                .await
                .map_err(|e| {
                    AthenasError::Download(format!("Failed to open chunk file for resume: {}", e))
                })?
        } else {
            tokio::fs::File::create(chunk_path).await.map_err(|e| {
                AthenasError::Download(format!("Failed to create chunk file: {}", e))
            })?
        };

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut write_buf: Vec<u8> = Vec::with_capacity(WRITE_BUFFER_SIZE);
        let mut stream_error: Option<String> = None;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    write_buf.extend_from_slice(&chunk);
                    downloaded.fetch_add(chunk.len() as u64, Ordering::Relaxed);

                    if write_buf.len() >= WRITE_BUFFER_SIZE {
                        if let Err(e) = writer.write_all(&write_buf).await {
                            stream_error = Some(format!("Write error: {}", e));
                            break;
                        }
                        write_buf.clear();
                    }
                }
                Err(e) => {
                    stream_error = Some(format!("Stream error: {}", e));
                    break;
                }
            }
        }

        // Flush whatever we have so far (important for resume)
        if !write_buf.is_empty() {
            if let Err(e) = writer.write_all(&write_buf).await {
                if stream_error.is_none() {
                    stream_error = Some(format!("Write error: {}", e));
                }
            }
        }
        let _ = writer.flush().await;

        if let Some(err) = stream_error {
            warn!(
                "Chunk {}-{} attempt {} stream error: {} (saved partial for resume)",
                start,
                end,
                attempt + 1,
                err
            );
            if attempt < MAX_RETRIES {
                let delay = RETRY_BASE_DELAY_MS * (1 << attempt);
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                continue;
            }
            // Final attempt failed — check if we got enough by chance
            let final_size = std::fs::metadata(chunk_path).map(|m| m.len()).unwrap_or(0);
            if final_size == expected_size {
                return Ok(());
            }
            return Err(AthenasError::Download(format!(
                "Chunk {}-{} failed after {} retries: {} (got {}/{} bytes)",
                start, end, MAX_RETRIES, err, final_size, expected_size
            )));
        }

        // Verify chunk size
        let final_size = std::fs::metadata(chunk_path).map(|m| m.len()).unwrap_or(0);
        if final_size != expected_size {
            warn!(
                "Chunk {}-{} size mismatch: got {} expected {} (attempt {})",
                start,
                end,
                final_size,
                expected_size,
                attempt + 1
            );
            if attempt < MAX_RETRIES {
                let delay = RETRY_BASE_DELAY_MS * (1 << attempt);
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                // Delete partial file to restart clean
                let _ = tokio::fs::remove_file(chunk_path).await;
                continue;
            }
            return Err(AthenasError::Download(format!(
                "Chunk {}-{} size mismatch after {} retries: got {} expected {}",
                start, end, MAX_RETRIES, final_size, expected_size
            )));
        }

        debug!(
            "Chunk bytes={}..{} ({} MB) complete",
            start,
            end,
            (end - start) / (1024 * 1024),
        );
        return Ok(());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_join_rejects_traversal_and_weird_paths() {
        let base = PathBuf::from("/tmp/models");
        assert!(safe_join(&base, "").is_err());
        assert!(safe_join(&base, "../evil").is_err());
        assert!(safe_join(&base, "a/../../b").is_err());
        assert!(safe_join(&base, "/abs/path").is_err());
        assert!(safe_join(&base, "\\abs").is_err());
        assert!(safe_join(&base, "a\\b").is_err());
        assert!(safe_join(&base, "./x").is_err());
        // Interior `.` is normalized away by components() — safe to accept.
        assert_eq!(safe_join(&base, "a/./b").unwrap(), base.join("a/b"));
    }

    #[test]
    fn safe_join_accepts_normal_relative_paths() {
        let base = PathBuf::from("/tmp/models");
        assert_eq!(
            safe_join(&base, "model.onnx").unwrap(),
            base.join("model.onnx")
        );
        assert_eq!(
            safe_join(&base, "onnx/model.onnx_data").unwrap(),
            base.join("onnx").join("model.onnx_data")
        );
    }
}
