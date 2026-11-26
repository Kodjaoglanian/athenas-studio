use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use athenas_core::{AthenasError, Result};

use crate::client::HuggingFaceClient;

const WRITE_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB write buffer
const MAX_CONCURRENT_CHUNKS: usize = 4; // Concurrent chunk downloads (gentle on slow/flaky networks)
const MAX_CHUNK_SIZE: u64 = 64 * 1024 * 1024; // 64 MB max chunk size so each chunk checkpoints quickly
const MIN_CHUNK_SIZE: u64 = 4 * 1024 * 1024; // 4 MB minimum chunk size
const MAX_CHUNK_RETRIES: usize = 8; // Max retries per chunk on transient error
const MAX_SINGLE_STREAM_RETRIES: usize = 10; // Max retries for single-stream download
const RETRY_BASE_DELAY_MS: u64 = 1000; // Base delay for exponential backoff
const RETRY_MAX_DELAY_MS: u64 = 30_000; // Cap backoff delay at 30s

/// Join a repo-relative remote path onto a local dir, rejecting anything
/// that would escape it (`..`, absolute paths, Windows separators/drives).
fn safe_join(base: &Path, rel: &str) -> Result<PathBuf> {
    if rel.is_empty() || rel.starts_with('/') || rel.starts_with('\\') || rel.contains('\\') {
        return Err(AthenasError::Download(format!("unsafe path: {rel}")));
    }
    let mut out = base.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            std::path::Component::Normal(c) => out.push(c),
            _ => return Err(AthenasError::Download(format!("unsafe path: {rel}"))),
        }
    }
    Ok(out)
}

/// Validate a HuggingFace `repo_id` (e.g. `org/model`) and return the safe
/// local directory name (`org__model`), rejecting path traversal.
fn safe_repo_dir_name(repo_id: &str) -> Result<String> {
    if repo_id.is_empty()
        || repo_id.starts_with('/')
        || repo_id.starts_with('\\')
        || repo_id.contains('\\')
    {
        return Err(AthenasError::Download(format!("unsafe repo_id: {repo_id}")));
    }
    let mut parts = Vec::new();
    for comp in Path::new(repo_id).components() {
        match comp {
            std::path::Component::Normal(c) => parts.push(c.to_string_lossy().into_owned()),
            _ => return Err(AthenasError::Download(format!("unsafe repo_id: {repo_id}"))),
        }
    }
    if parts.is_empty() {
        return Err(AthenasError::Download(format!("unsafe repo_id: {repo_id}")));
    }
    Ok(parts.join("__"))
}

/// Build the `.part` temporary path by appending `.part` to the full filename
/// (so `model.onnx` -> `model.onnx.part` and `model.onnx_data` -> `model.onnx_data.part`
/// never collide).
fn part_path_for(file_path: &Path) -> PathBuf {
    let mut os = file_path.as_os_str().to_os_string();
    os.push(".part");
    PathBuf::from(os)
}

/// Build a per-file isolated chunk temp path in the destination's parent directory.
/// Includes a sanitized filename prefix and a short hash so concurrent or leftover
/// downloads in the same repo directory can never overwrite each other's chunks.
fn chunk_path_for(file_path: &Path, chunk_index: u64) -> PathBuf {
    use sha2::Digest;

    let dir = file_path.parent().unwrap_or_else(|| Path::new("."));
    let fname = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    let safe_name: String = fname
        .chars()
        .take(48)
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let digest = sha2::Sha256::digest(fname.as_bytes());
    let hash_prefix = hex::encode(&digest[..4]);
    dir.join(format!(
        ".athenas_chunk_{}_{}_{}",
        safe_name, hash_prefix, chunk_index
    ))
}

fn retry_delay(attempt: usize) -> std::time::Duration {
    let shift = (attempt as u32).min(10);
    let ms = RETRY_BASE_DELAY_MS
        .saturating_mul(1u64 << shift)
        .min(RETRY_MAX_DELAY_MS);
    std::time::Duration::from_millis(ms)
}

fn parse_content_range_total(header_val: &str) -> Option<u64> {
    header_val.split('/').nth(1)?.trim().parse::<u64>().ok()
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
}

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
        let safe_repo = safe_repo_dir_name(repo_id)?;
        let model_dir = safe_join(&self.models_dir, &safe_repo)?;
        tokio::fs::create_dir_all(&model_dir).await?;

        // Filenames may carry a subdir (e.g. `onnx/model.onnx`) — keep the
        // structure but never allow escaping the model dir.
        let file_path = safe_join(&model_dir, filename)?;
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
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
        if tokio::fs::try_exists(file_path).await.unwrap_or(false) {
            info!("File already exists: {}", file_path.display());
            self.verify_sha256(file_path, expected_sha256.as_deref())
                .await?;
            return Ok(file_path.clone());
        }

        // Single Range probe to check Range support and discover total size
        match self.probe_range_support(&download_url).await {
            Ok(Some(0)) => {
                // Short-circuit 0-byte files (avoids end - 1 underflow in Range math)
                tokio::fs::File::create(file_path).await.map_err(|e| {
                    AthenasError::Download(format!("Failed to create empty file: {}", e))
                })?;
                if let Some(ref tx) = progress_tx {
                    let _ = tx
                        .send(DownloadProgress {
                            downloaded_bytes: 0,
                            total_bytes: Some(0),
                            speed_mbps: 0.0,
                            percent: Some(100.0),
                        })
                        .await;
                }
                self.verify_sha256(file_path, expected_sha256.as_deref())
                    .await?;
                return Ok(file_path.clone());
            }
            Ok(Some(total_bytes)) => {
                info!(
                    "Server supports range requests ({:.1} MB), using resilient chunked download",
                    total_bytes as f64 / (1024.0 * 1024.0)
                );
                match self
                    .parallel_download(&download_url, file_path, total_bytes, progress_tx.clone())
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
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!("Range probe failed ({}), falling back to single-stream", e);
            }
        }

        info!("Using resilient single-stream download");
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
        let safe_repo = safe_repo_dir_name(repo_id)?;
        let variant_dir = variant_dir.trim_matches('/');
        if !variant_dir.is_empty() {
            // Validate variant_dir against traversal before using it.
            let _ = safe_join(&self.models_dir, variant_dir)?;
        }

        let files = self
            .client
            .get_model_files_recursive(repo_id, revision)
            .await?;

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
        // Skip when `prefix.is_empty()` because `variant_files` already contains
        // all root-level files. When `prefix` is non-empty, compare against the
        // variant-relative path so root files never overwrite variant-specific
        // configs like `genai_config.json`.
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
        let root_support: Vec<_> = if prefix.is_empty() {
            Vec::new()
        } else {
            files
                .iter()
                .filter(|f| {
                    f.r#type == "file"
                        && !f.path.contains('/')
                        && SUPPORT.contains(&f.path.as_str())
                })
                .filter(|f| {
                    !variant_files
                        .iter()
                        .any(|v| v.path.strip_prefix(&prefix) == Some(f.path.as_str()))
                })
                .collect()
        };

        let repo_root = safe_join(&self.models_dir, &safe_repo)?;
        let dest_dir = if variant_dir.is_empty() {
            repo_root
        } else {
            let variant_name = variant_dir.rsplit('/').next().unwrap_or(variant_dir);
            safe_join(&repo_root, variant_name)?
        };
        tokio::fs::create_dir_all(&dest_dir).await?;

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
                tokio::fs::create_dir_all(parent).await?;
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
    async fn verify_sha256(&self, path: &Path, expected: Option<&str>) -> Result<()> {
        let Some(expected) = expected else {
            return Ok(());
        };
        let expected = expected.to_lowercase();
        let hash_path = path.to_path_buf();
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
        let _ = tokio::fs::remove_file(path).await;
        Err(AthenasError::Download(format!(
            "SHA256 mismatch: expected {}, got {}",
            expected, actual
        )))
    }

    /// Single Range probe (`bytes=0-0`) to detect Range support and return
    /// total file size without issuing a duplicate probe.
    async fn probe_range_support(&self, url: &str) -> Result<Option<u64>> {
        let client = self.client.download_client().clone();
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

        if status == reqwest::StatusCode::PARTIAL_CONTENT {
            if let Some(total) = resp
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_content_range_total)
            {
                return Ok(Some(total));
            }
            warn!("Range probe returned 206 without parseable Content-Range");
            return Ok(None);
        }

        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            // A 0-byte file responds with 416 and `Content-Range: bytes */0`.
            if let Some(0) = resp
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_content_range_total)
            {
                return Ok(Some(0));
            }
        }

        warn!(
            "Server does not support Range requests (status: {}), falling back to single-stream",
            status
        );
        Ok(None)
    }

    async fn parallel_download(
        &self,
        url: &str,
        file_path: &PathBuf,
        total_bytes: u64,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    ) -> Result<PathBuf> {
        let client = self.client.download_client().clone();
        let token = self.client.token().map(|t| t.to_string());

        // Note: We intentionally do NOT use the CDN URL from the redirect for chunk
        // downloads. HuggingFace LFS CDN signed URLs reject arbitrary range requests
        // with 403 "invalid range". Instead, each chunk request goes to the original
        // URL and gets a fresh redirect with a signed URL for that specific range.

        info!(
            "File size: {} bytes ({:.1} MB)",
            total_bytes,
            total_bytes as f64 / (1024.0 * 1024.0)
        );

        // Adaptive chunk sizing: cap each chunk at MAX_CHUNK_SIZE (64 MB) so
        // slow/flaky connections make steady checkpointed progress instead of
        // risking giant ~1 GB chunks. Limit active connections to MAX_CONCURRENT_CHUNKS.
        let min_chunks_by_max_size = total_bytes.div_ceil(MAX_CHUNK_SIZE).max(1);
        let desired_parallel =
            (total_bytes / MIN_CHUNK_SIZE).clamp(1, MAX_CONCURRENT_CHUNKS as u64);
        let num_chunks = min_chunks_by_max_size.max(desired_parallel);
        let chunk_size = total_bytes.div_ceil(num_chunks);

        info!(
            "Using {} chunks of ~{:.1} MB each (max {} concurrent)",
            num_chunks,
            chunk_size as f64 / (1024.0 * 1024.0),
            MAX_CONCURRENT_CHUNKS
        );

        // Shared progress counter (display only; completion verified from disk)
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

                    let current = progress_downloaded.load(Ordering::Relaxed).min(total);
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

        let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_CHUNKS));
        let mut tasks = Vec::new();
        let mut chunk_specs: Vec<(PathBuf, u64)> = Vec::new();

        for i in 0..num_chunks {
            let start = i * chunk_size;
            if start >= total_bytes {
                break;
            }
            let end = (start + chunk_size).min(total_bytes);
            let expected_size = end - start;

            let chunk_path = chunk_path_for(file_path, i);
            chunk_specs.push((chunk_path.clone(), expected_size));

            // Resume support: check existing chunk size asynchronously
            if let Ok(meta) = tokio::fs::metadata(&chunk_path).await {
                let len = meta.len();
                if len == expected_size {
                    debug!(
                        "Chunk {} already complete ({} bytes), skipping",
                        i, expected_size
                    );
                    downloaded.fetch_add(expected_size, Ordering::Relaxed);
                    continue;
                } else if len > expected_size {
                    let _ = tokio::fs::remove_file(&chunk_path).await;
                } else if len > 0 {
                    downloaded.fetch_add(len, Ordering::Relaxed);
                }
            }

            let chunk_url = url.to_string();
            let client = client.clone();
            let downloaded = downloaded.clone();
            let token = token.clone();
            let sem = semaphore.clone();

            tasks.push(tokio::spawn(async move {
                let _permit = sem
                    .acquire_owned()
                    .await
                    .map_err(|e| AthenasError::Download(format!("Semaphore closed: {}", e)))?;
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
            // Preserve valid completed and partial chunks on disk for resume;
            // only remove oversized/corrupt chunks.
            for (p, expected) in &chunk_specs {
                if let Ok(meta) = tokio::fs::metadata(p).await {
                    if meta.len() > *expected {
                        let _ = tokio::fs::remove_file(p).await;
                    }
                }
            }
            return Err(AthenasError::Download(format!(
                "Parallel download failed (re-run to resume): {}",
                errors.join("; ")
            )));
        }

        // Verify completion by inspecting actual chunk file sizes on disk
        // (immune to progress counter drift on retries).
        let mut total_on_disk: u64 = 0;
        for (p, expected) in &chunk_specs {
            let actual = tokio::fs::metadata(p).await.map(|m| m.len()).unwrap_or(0);
            if actual != *expected {
                if actual > *expected {
                    let _ = tokio::fs::remove_file(p).await;
                }
                return Err(AthenasError::Download(format!(
                    "Chunk {} size mismatch on disk: {} / {} bytes (re-run to resume)",
                    p.display(),
                    actual,
                    expected
                )));
            }
            total_on_disk = total_on_disk.saturating_add(actual);
        }

        if total_on_disk != total_bytes {
            return Err(AthenasError::Download(format!(
                "Download incomplete on disk: {} / {} bytes (re-run to resume)",
                total_on_disk, total_bytes
            )));
        }

        // Concatenate chunk files into final file
        info!("Concatenating {} chunk files...", chunk_specs.len());
        let temp_path = part_path_for(file_path);
        {
            let mut out = tokio::fs::File::create(&temp_path).await.map_err(|e| {
                AthenasError::Download(format!("Failed to create output file: {}", e))
            })?;

            for (p, _) in &chunk_specs {
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
        for (p, _) in &chunk_specs {
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

        let elapsed = start_time.elapsed().as_secs_f64();
        info!(
            "Downloaded {} ({} bytes) in {:.1}s ({:.1} MB/s avg)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            total_bytes,
            elapsed,
            if elapsed > 0.0 {
                (total_bytes as f64 / (1024.0 * 1024.0)) / elapsed
            } else {
                0.0
            }
        );
        Ok(file_path.clone())
    }

    async fn single_stream_download(
        &self,
        url: &str,
        file_path: &PathBuf,
        progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    ) -> Result<PathBuf> {
        use futures::StreamExt;

        let client = self.client.download_client().clone();
        let temp_path = part_path_for(file_path);
        let start_time = std::time::Instant::now();
        let mut known_total: Option<u64> = None;

        for attempt in 0..=MAX_SINGLE_STREAM_RETRIES {
            let mut existing_bytes = tokio::fs::metadata(&temp_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);

            if let Some(total) = known_total {
                if existing_bytes == total {
                    break;
                }
                if existing_bytes > total {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    existing_bytes = 0;
                }
            }

            let mut req = client.get(url);
            if let Some(token) = self.client.token() {
                req = req.bearer_auth(token);
            }
            if existing_bytes > 0 {
                info!(
                    "Resuming single-stream download from byte {} ({:.1} MB) [attempt {}/{}]",
                    existing_bytes,
                    existing_bytes as f64 / (1024.0 * 1024.0),
                    attempt + 1,
                    MAX_SINGLE_STREAM_RETRIES + 1
                );
                req = req.header("Range", format!("bytes={}-", existing_bytes));
            }

            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        "Single-stream attempt {} request failed: {}",
                        attempt + 1,
                        e
                    );
                    if attempt < MAX_SINGLE_STREAM_RETRIES {
                        tokio::time::sleep(retry_delay(attempt)).await;
                        continue;
                    }
                    return Err(AthenasError::Download(format!(
                        "Download request failed after {} retries: {}",
                        MAX_SINGLE_STREAM_RETRIES, e
                    )));
                }
            };

            let status = resp.status();

            // Handle 416 Range Not Satisfiable (either already complete or stale .part)
            if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                if let Some(total) = resp
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .and_then(parse_content_range_total)
                {
                    known_total = Some(total);
                    if existing_bytes == total {
                        info!("Partial file already complete ({} bytes)", total);
                        break;
                    }
                }
                warn!(
                    "HTTP 416 on single-stream resume at {} bytes — restarting cleanly",
                    existing_bytes
                );
                let _ = tokio::fs::remove_file(&temp_path).await;
                if attempt < MAX_SINGLE_STREAM_RETRIES {
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                return Err(AthenasError::Download(
                    "Download failed: HTTP 416 Range Not Satisfiable".into(),
                ));
            }

            let is_resume = status == reqwest::StatusCode::PARTIAL_CONTENT;
            if !status.is_success() && !is_resume {
                let text = resp.text().await.unwrap_or_default();
                if is_retryable_status(status) && attempt < MAX_SINGLE_STREAM_RETRIES {
                    warn!(
                        "Single-stream attempt {} returned {}, retrying: {}",
                        attempt + 1,
                        status,
                        text
                    );
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                return Err(AthenasError::Download(format!(
                    "Download failed: {} - {}",
                    status, text
                )));
            }

            // Update expected total size if provided by the server
            let resp_total = if is_resume {
                resp.headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .and_then(parse_content_range_total)
                    .or_else(|| resp.content_length().map(|cl| existing_bytes + cl))
            } else {
                resp.content_length()
            };
            if let Some(t) = resp_total {
                known_total = Some(t);
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

            let mut stream = resp.bytes_stream();
            let mut downloaded: u64 = initial_bytes;
            let mut last_update = std::time::Instant::now();
            let mut write_buf: Vec<u8> = Vec::with_capacity(WRITE_BUFFER_SIZE);
            let mut speed_window_bytes: u64 = 0;
            let mut speed_window_start = std::time::Instant::now();
            let mut stream_err: Option<String> = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        write_buf.extend_from_slice(&chunk);
                        downloaded += chunk.len() as u64;
                        speed_window_bytes += chunk.len() as u64;

                        if write_buf.len() >= WRITE_BUFFER_SIZE {
                            if let Err(e) = file.write_all(&write_buf).await {
                                stream_err = Some(format!("Write error: {}", e));
                                break;
                            }
                            write_buf.clear();
                        }

                        if let Some(ref tx) = progress_tx {
                            let now = std::time::Instant::now();
                            if now.duration_since(last_update)
                                > std::time::Duration::from_millis(200)
                            {
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
                                let percent = known_total
                                    .map(|t| (downloaded as f64 / t.max(1) as f64) * 100.0);
                                let _ = tx
                                    .send(DownloadProgress {
                                        downloaded_bytes: downloaded,
                                        total_bytes: known_total,
                                        speed_mbps,
                                        percent,
                                    })
                                    .await;
                            }
                        }
                    }
                    Err(e) => {
                        stream_err = Some(format!("Stream error: {}", e));
                        break;
                    }
                }
            }

            // Flush buffered bytes so resume picks up right where we stopped
            if !write_buf.is_empty() {
                if let Err(e) = file.write_all(&write_buf).await {
                    if stream_err.is_none() {
                        stream_err = Some(format!("Write error: {}", e));
                    }
                }
            }
            let _ = file.flush().await;
            drop(file);

            let on_disk = tokio::fs::metadata(&temp_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);

            if let Some(err) = stream_err {
                warn!(
                    "Single-stream attempt {} interrupted at {} bytes: {} (saved .part for resume)",
                    attempt + 1,
                    on_disk,
                    err
                );
                if let Some(total) = known_total {
                    if on_disk == total {
                        break;
                    }
                }
                if attempt < MAX_SINGLE_STREAM_RETRIES {
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                return Err(AthenasError::Download(format!(
                    "Single-stream download failed after {} retries: {}",
                    MAX_SINGLE_STREAM_RETRIES, err
                )));
            }

            // Stream ended cleanly — verify against known_total if available
            if let Some(total) = known_total {
                if on_disk == total {
                    break;
                }
                warn!(
                    "Single-stream ended early: {} / {} bytes (attempt {})",
                    on_disk,
                    total,
                    attempt + 1
                );
                if on_disk > total {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                }
                if attempt < MAX_SINGLE_STREAM_RETRIES {
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                return Err(AthenasError::Download(format!(
                    "Download incomplete: {} / {} bytes after {} retries",
                    on_disk, total, MAX_SINGLE_STREAM_RETRIES
                )));
            }

            // No Content-Length/Content-Range was provided and stream completed cleanly
            break;
        }

        // Final verification before renaming `.part` to destination
        let final_size = tokio::fs::metadata(&temp_path)
            .await
            .map(|m| m.len())
            .map_err(|e| AthenasError::Download(format!("Partial file missing: {}", e)))?;
        if let Some(total) = known_total {
            if final_size != total {
                return Err(AthenasError::Download(format!(
                    "Download size mismatch before rename: {} / {} bytes",
                    final_size, total
                )));
            }
        }

        tokio::fs::rename(&temp_path, file_path)
            .await
            .map_err(|e| AthenasError::Download(format!("Rename error: {}", e)))?;

        if let Some(ref tx) = progress_tx {
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed_mbps = if elapsed > 0.0 {
                (final_size as f64 / (1024.0 * 1024.0)) / elapsed
            } else {
                0.0
            };
            let _ = tx
                .send(DownloadProgress {
                    downloaded_bytes: final_size,
                    total_bytes: Some(final_size),
                    speed_mbps,
                    percent: Some(100.0),
                })
                .await;
        }

        info!(
            "Downloaded {} ({} bytes)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            final_size
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
            .filter(|f| {
                f.r#type == "file"
                    && Path::new(&f.path)
                        .extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            })
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

/// Download a chunk with automatic retry and per-chunk resume on transient errors.
/// Never deletes valid partial chunk bytes on network/timeout/stream errors;
/// only resets the chunk file on HTTP 416 or if on-disk size exceeds `expected_size`.
async fn download_chunk_with_retry(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    start: u64,
    end: u64,
    chunk_path: &PathBuf,
    downloaded: Arc<AtomicU64>,
) -> Result<()> {
    use futures::StreamExt;

    let expected_size = end.saturating_sub(start);
    if expected_size == 0 {
        tokio::fs::File::create(chunk_path).await.map_err(|e| {
            AthenasError::Download(format!("Failed to create empty chunk file: {}", e))
        })?;
        return Ok(());
    }

    for attempt in 0..=MAX_CHUNK_RETRIES {
        let mut chunk_have = tokio::fs::metadata(chunk_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);

        if chunk_have == expected_size {
            debug!("Chunk {}-{} already complete, skipping", start, end);
            return Ok(());
        }
        if chunk_have > expected_size {
            warn!(
                "Chunk {}-{} oversized ({} > {}), resetting",
                start, end, chunk_have, expected_size
            );
            let _ = tokio::fs::remove_file(chunk_path).await;
            chunk_have = 0;
        }

        let resume_from = start + chunk_have;
        let range = format!("bytes={}-{}", resume_from, end - 1);
        if attempt > 0 {
            info!(
                "Retrying chunk {}-{} from byte {} (have {}/{} bytes, attempt {}/{})",
                start,
                end,
                resume_from,
                chunk_have,
                expected_size,
                attempt + 1,
                MAX_CHUNK_RETRIES + 1
            );
        }

        let mut req = client.get(url).header("Range", &range);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Chunk {}-{} attempt {} request failed: {} (preserving {} partial bytes)",
                    start,
                    end,
                    attempt + 1,
                    e,
                    chunk_have
                );
                if attempt < MAX_CHUNK_RETRIES {
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                return Err(AthenasError::Download(format!(
                    "Chunk {}-{} request failed after {} retries: {}",
                    start, end, MAX_CHUNK_RETRIES, e
                )));
            }
        };

        let status = resp.status();
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            warn!(
                "Chunk {}-{} attempt {} got HTTP 416 — resetting chunk",
                start,
                end,
                attempt + 1
            );
            let _ = tokio::fs::remove_file(chunk_path).await;
            if attempt < MAX_CHUNK_RETRIES {
                tokio::time::sleep(retry_delay(attempt)).await;
                continue;
            }
            return Err(AthenasError::Download(format!(
                "Chunk {}-{} failed with HTTP 416 after {} retries",
                start, end, MAX_CHUNK_RETRIES
            )));
        }

        if status != reqwest::StatusCode::PARTIAL_CONTENT {
            let text = resp.text().await.unwrap_or_default();
            warn!(
                "Chunk {}-{} attempt {} unexpected status: {} {}",
                start,
                end,
                attempt + 1,
                status,
                text
            );
            // Preserve valid partial bytes; retry transient errors.
            if (is_retryable_status(status) || status == reqwest::StatusCode::OK)
                && attempt < MAX_CHUNK_RETRIES
            {
                tokio::time::sleep(retry_delay(attempt)).await;
                continue;
            }
            return Err(AthenasError::Download(format!(
                "Chunk {}-{} download failed: {} - {}",
                start, end, status, text
            )));
        }

        // Open file for append if resuming, otherwise create fresh
        let mut writer = if chunk_have > 0 {
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

        // Flush whatever we have so far so the next retry resumes from here
        if !write_buf.is_empty() {
            if let Err(e) = writer.write_all(&write_buf).await {
                if stream_error.is_none() {
                    stream_error = Some(format!("Write error: {}", e));
                }
            }
        }
        let _ = writer.flush().await;
        drop(writer);

        let final_size = tokio::fs::metadata(chunk_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);

        if let Some(err) = stream_error {
            if final_size == expected_size {
                return Ok(());
            }
            warn!(
                "Chunk {}-{} attempt {} stream error at {}/{} bytes: {} (saved partial for resume)",
                start,
                end,
                attempt + 1,
                final_size,
                expected_size,
                err
            );
            if final_size > expected_size {
                let _ = tokio::fs::remove_file(chunk_path).await;
            }
            if attempt < MAX_CHUNK_RETRIES {
                tokio::time::sleep(retry_delay(attempt)).await;
                continue;
            }
            return Err(AthenasError::Download(format!(
                "Chunk {}-{} failed after {} retries: {} (got {}/{} bytes)",
                start, end, MAX_CHUNK_RETRIES, err, final_size, expected_size
            )));
        }

        if final_size == expected_size {
            debug!(
                "Chunk bytes={}..{} ({} MB) complete",
                start,
                end,
                expected_size / (1024 * 1024),
            );
            return Ok(());
        }

        // Stream closed early without error: preserve partial bytes if < expected,
        // reset only if > expected.
        warn!(
            "Chunk {}-{} incomplete after EOF: got {} expected {} (attempt {})",
            start,
            end,
            final_size,
            expected_size,
            attempt + 1
        );
        if final_size > expected_size {
            let _ = tokio::fs::remove_file(chunk_path).await;
        }
        if attempt < MAX_CHUNK_RETRIES {
            tokio::time::sleep(retry_delay(attempt)).await;
            continue;
        }
        return Err(AthenasError::Download(format!(
            "Chunk {}-{} size mismatch after {} retries: got {} expected {}",
            start, end, MAX_CHUNK_RETRIES, final_size, expected_size
        )));
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

    #[test]
    fn safe_repo_dir_name_rejects_traversal() {
        assert_eq!(safe_repo_dir_name("org/repo").unwrap(), "org__repo");
        assert!(safe_repo_dir_name("").is_err());
        assert!(safe_repo_dir_name("../evil").is_err());
        assert!(safe_repo_dir_name("/abs/repo").is_err());
        assert!(safe_repo_dir_name("org\\repo").is_err());
    }

    #[test]
    fn part_path_does_not_collide_for_onnx_and_onnx_data() {
        let dir = PathBuf::from("/tmp/models/org__repo");
        let onnx = part_path_for(&dir.join("model.onnx"));
        let onnx_data = part_path_for(&dir.join("model.onnx_data"));
        assert_eq!(onnx, dir.join("model.onnx.part"));
        assert_eq!(onnx_data, dir.join("model.onnx_data.part"));
        assert_ne!(onnx, onnx_data);
    }

    #[test]
    fn chunk_paths_are_isolated_per_filename() {
        let dir = PathBuf::from("/tmp/models/org__repo");
        let c1 = chunk_path_for(&dir.join("model-q4.gguf"), 0);
        let c2 = chunk_path_for(&dir.join("model-q8.gguf"), 0);
        let c3 = chunk_path_for(&dir.join("a b.gguf"), 0);
        let c4 = chunk_path_for(&dir.join("a_b.gguf"), 0);
        assert_ne!(c1, c2);
        assert_ne!(c3, c4);
        assert_eq!(c1.parent(), Some(dir.as_path()));
    }
}
