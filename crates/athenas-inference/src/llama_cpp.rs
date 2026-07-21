use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::info;

use athenas_core::{AthenasError, HardwareInfo, Result};

use crate::backend::{Backend, ModelInfo};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, InferenceStats,
    ModelLoadConfig, Role, StreamChunk,
};

/// llama.cpp backend — uses llama.cpp server subprocess for inference
pub struct LlamaCppBackend {
    hardware: HardwareInfo,
    loaded: bool,
    model_path: String,
    model_name: String,
    context_size: u32,
    gpu_layers: i32,
    server_handle: Option<tokio::process::Child>,
    server_port: u16,
    client: reqwest::Client,
}

impl LlamaCppBackend {
    pub fn new(hardware: &HardwareInfo) -> Self {
        Self {
            hardware: hardware.clone(),
            loaded: false,
            model_path: String::new(),
            model_name: String::new(),
            context_size: 4096,
            gpu_layers: -1,
            server_handle: None,
            server_port: 0,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap(),
        }
    }

    fn find_llama_server(&self) -> Option<String> {
        // 1. Check PATH
        for cmd in &["llama-server", "llama_server", "server"] {
            if which::which(cmd).is_ok() {
                return Some(cmd.to_string());
            }
        }

        // 2. Check common install locations
        let home = std::env::var("HOME").unwrap_or_default();
        let candidates = [
            format!("{}/.athenas/bin/llama-server", home),
            format!("{}/.local/bin/llama-server", home),
            "/usr/local/bin/llama-server".to_string(),
            "/usr/bin/llama-server".to_string(),
            "/opt/llama.cpp/build/bin/llama-server".to_string(),
        ];

        for path in &candidates {
            if std::path::Path::new(path).exists() {
                return Some(path.clone());
            }
        }

        None
    }

    async fn start_server(&mut self, config: &ModelLoadConfig) -> Result<()> {
        let server_bin = if let Some(path) = self.find_llama_server() {
            // Validate: if it's in ~/.athenas/bin, check for shared libs
            let p = std::path::Path::new(&path);
            if let Some(parent) = p.parent() {
                let needs_lib = std::env::consts::OS == "linux" || std::env::consts::OS == "macos";
                let has_lib = parent.join("libllama-server-impl.so").exists()
                    || parent.join("libllama-server-impl.dylib").exists();
                if needs_lib && !has_lib && path.contains(".athenas") {
                    info!("llama-server found but shared libs missing, re-downloading...");
                    let _ = std::fs::remove_file(&path);
                    let new_path = crate::backend_setup::ensure_llama_server().await?;
                    new_path.to_string_lossy().to_string()
                } else {
                    path
                }
            } else {
                path
            }
        } else {
            info!("llama-server not found, auto-downloading...");
            let path = crate::backend_setup::ensure_llama_server().await?;
            path.to_string_lossy().to_string()
        };

        self.server_port = find_free_port();

        let mut cmd = tokio::process::Command::new(&server_bin);

        // Set LD_LIBRARY_PATH to the directory containing llama-server
        // so it can find shared libraries (libllama-server-impl.so, etc.)
        if let Some(parent) = std::path::Path::new(&server_bin).parent() {
            let lib_path = parent.to_string_lossy().to_string();
            let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            let new_ld_path = if existing.is_empty() {
                lib_path
            } else {
                format!("{}:{}", lib_path, existing)
            };
            cmd.env("LD_LIBRARY_PATH", new_ld_path);
        }

        cmd.arg("--model")
            .arg(&config.model_path)
            .arg("--ctx-size")
            .arg(config.context_size.to_string())
            .arg("--batch-size")
            .arg(config.batch_size.to_string())
            .arg("--threads")
            .arg(config.threads.to_string())
            .arg("--port")
            .arg(self.server_port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--parallel")
            .arg("1");

        if config.gpu_layers >= 0 {
            cmd.arg("--n-gpu-layers").arg(config.gpu_layers.to_string());
        } else if self.hardware.has_cuda || self.hardware.has_rocm {
            cmd.arg("--n-gpu-layers").arg("999");
        }

        if config.flash_attention {
            cmd.arg("--flash-attn").arg("on");
        }

        if config.use_mmap {
            cmd.arg("--mmap");
        }

        if config.use_mlock {
            cmd.arg("--mlock");
        }

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        info!(
            "Starting llama-server on port {} with model: {}",
            self.server_port, config.model_path
        );

        let child = cmd
            .spawn()
            .map_err(|e| AthenasError::Backend(format!("Failed to start llama-server: {}", e)))?;

        self.server_handle = Some(child);

        // Wait for server to be ready
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        for _attempt in 0..20 {
            // Check if process exited early
            if let Some(ref mut child) = self.server_handle {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        // Try to read stderr for diagnostic info
                        let stderr_msg = if let Some(stderr) = child.stderr.take() {
                            use tokio::io::AsyncReadExt;
                            let mut buf = Vec::new();
                            let mut stderr = stderr;
                            let _ = stderr.read_to_end(&mut buf).await;
                            String::from_utf8_lossy(&buf).to_string()
                        } else {
                            String::new()
                        };

                        let mut msg = format!("llama-server exited early with status: {}", status);
                        if !stderr_msg.is_empty() {
                            msg.push_str(&format!("\nstderr: {}", stderr_msg));
                        }
                        if status.code() == Some(127) {
                            msg.push_str(
                                "\n\nHint: exit code 127 usually means the binary has missing \
                                 shared libraries. Try running 'ldd <path>' to check.\n\
                                 On Ubuntu/Debian: apt install -y libgomp1",
                            );
                        }
                        return Err(AthenasError::Backend(msg));
                    }
                    Ok(None) => {}
                    Err(_) => {}
                }
            }

            // Try health check with timeout
            let url = format!("http://127.0.0.1:{}/health", self.server_port);
            let health_req = self
                .client
                .get(&url)
                .timeout(tokio::time::Duration::from_secs(2));
            if let Ok(resp) = health_req.send().await {
                if resp.status().is_success() {
                    info!("llama-server is ready on port {}", self.server_port);
                    return Ok(());
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        // Kill the server if it didn't start in time
        if let Some(ref mut child) = self.server_handle {
            let _ = child.kill().await;
        }
        self.server_handle = None;

        Err(AthenasError::Backend(
            "llama-server failed to start within 10 seconds".to_string(),
        ))
    }

    async fn stop_server(&mut self) -> Result<()> {
        if let Some(mut child) = self.server_handle.take() {
            child.kill().await.map_err(|e| {
                AthenasError::Backend(format!("Failed to kill llama-server: {}", e))
            })?;
            info!("llama-server stopped");
        }
        Ok(())
    }

    fn server_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.server_port)
    }

    fn build_chat_prompt(&self, request: &ChatRequest) -> String {
        let mut prompt = String::new();
        for msg in &request.messages {
            match msg.role {
                Role::System => prompt.push_str(&format!("<|system|>\n{}\n", msg.content)),
                Role::User => prompt.push_str(&format!("<|user|>\n{}\n", msg.content)),
                Role::Assistant => prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content)),
                Role::Tool => prompt.push_str(&format!("<|tool|>\n{}\n", msg.content)),
            }
        }
        prompt.push_str("<|assistant|>\n");
        prompt
    }
}

#[async_trait]
impl Backend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    async fn load_model(&mut self, config: ModelLoadConfig) -> Result<()> {
        self.model_path = config.model_path.clone();
        self.model_name = std::path::Path::new(&config.model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        self.context_size = config.context_size;
        self.gpu_layers = config.gpu_layers;

        self.start_server(&config).await?;
        self.loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.stop_server().await?;
        self.loaded = false;
        self.model_path.clear();
        self.model_name.clear();
        Ok(())
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        let prompt = self.build_chat_prompt(&request);

        let body = serde_json::json!({
            "prompt": prompt,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "n_predict": request.max_tokens.unwrap_or(2048),
            "stream": false,
            "stop": request.stop.as_deref().unwrap_or(&[]),
        });

        let url = format!("{}/completion", self.server_url());
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!(
                "llama-server returned {}: {}",
                status, text
            )));
        }

        let result: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AthenasError::Backend(format!("Invalid response: {}", e)))?;

        let content = result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let tokens_predicted = result
            .get("tokens_predicted")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let tokens_decoded = result
            .get("tokens_decoded")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let timings = result.get("timings");
        let time_total_ms = timings
            .and_then(|t| t.get("predicted_ms"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let tps = timings
            .and_then(|t| t.get("tokens_per_second"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let stats = InferenceStats {
            tokens_generated: tokens_decoded,
            tokens_prompt: tokens_predicted,
            time_total_ms,
            tokens_per_second: tps,
        };

        Ok(ChatResponse {
            model: self.model_name.clone(),
            message: ChatMessage::assistant(content),
            stats,
        })
    }

    async fn chat_stream(&self, request: ChatRequest, tx: mpsc::Sender<StreamChunk>) -> Result<()> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        let prompt = self.build_chat_prompt(&request);

        let body = serde_json::json!({
            "prompt": prompt,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "n_predict": request.max_tokens.unwrap_or(2048),
            "stream": true,
            "stop": request.stop.as_deref().unwrap_or(&[]),
        });

        let url = format!("{}/completion", self.server_url());
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!(
                "llama-server returned {}: {}",
                status, text
            )));
        }

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut full_text = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk =
                chunk_result.map_err(|e| AthenasError::Backend(format!("Stream error: {}", e)))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    let _ = tx
                        .send(StreamChunk {
                            text: String::new(),
                            done: true,
                            stats: Some(InferenceStats {
                                tokens_generated: full_text.split_whitespace().count() as u32,
                                tokens_prompt: 0,
                                time_total_ms: 0,
                                tokens_per_second: 0.0,
                            }),
                        })
                        .await;
                    return Ok(());
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let content = json.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if !content.is_empty() {
                        full_text.push_str(content);
                        let _ = tx
                            .send(StreamChunk {
                                text: content.to_string(),
                                done: false,
                                stats: None,
                            })
                            .await;
                    }

                    let stop = json.get("stop").and_then(|v| v.as_bool()).unwrap_or(false);
                    if stop {
                        let timings = json.get("timings");
                        let tps = timings
                            .and_then(|t| t.get("tokens_per_second"))
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        let _ = tx
                            .send(StreamChunk {
                                text: String::new(),
                                done: true,
                                stats: Some(InferenceStats {
                                    tokens_generated: full_text.split_whitespace().count() as u32,
                                    tokens_prompt: 0,
                                    time_total_ms: 0,
                                    tokens_per_second: tps,
                                }),
                            })
                            .await;
                        return Ok(());
                    }
                }
            }
        }

        let _ = tx
            .send(StreamChunk {
                text: String::new(),
                done: true,
                stats: None,
            })
            .await;
        Ok(())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        let body = serde_json::json!({
            "prompt": request.prompt,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "n_predict": request.max_tokens.unwrap_or(2048),
            "stream": false,
            "stop": request.stop.as_deref().unwrap_or(&[]),
        });

        let url = format!("{}/completion", self.server_url());
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!(
                "llama-server returned {}: {}",
                status, text
            )));
        }

        let result: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AthenasError::Backend(format!("Invalid response: {}", e)))?;

        let content = result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let tokens_decoded = result
            .get("tokens_decoded")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let tokens_predicted = result
            .get("tokens_predicted")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let timings = result.get("timings");
        let tps = timings
            .and_then(|t| t.get("tokens_per_second"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let time_ms = timings
            .and_then(|t| t.get("predicted_ms"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        Ok(CompletionResponse {
            model: self.model_name.clone(),
            text: content,
            stats: InferenceStats {
                tokens_generated: tokens_decoded,
                tokens_prompt: tokens_predicted,
                time_total_ms: time_ms,
                tokens_per_second: tps,
            },
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        let chat_req = ChatRequest {
            model: request.model.clone(),
            messages: vec![ChatMessage::user(&request.prompt)],
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            stream: true,
            stop: request.stop.clone(),
            ..Default::default()
        };
        self.chat_stream(chat_req, tx).await
    }

    fn model_info(&self) -> Option<ModelInfo> {
        if self.loaded {
            Some(ModelInfo {
                name: self.model_name.clone(),
                context_size: self.context_size,
                gpu_layers: self.gpu_layers,
                backend_name: "llama.cpp".to_string(),
            })
        } else {
            None
        }
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        if let Some(mut child) = self.server_handle.take() {
            let _ = child.start_kill();
        }
    }
}

fn find_free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| {
            let addr = listener.local_addr()?;
            Ok(addr.port())
        })
        .unwrap_or(9090)
}
