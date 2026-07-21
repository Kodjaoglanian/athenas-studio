use async_trait::async_trait;
use std::process::Stdio;
use tokio::sync::mpsc;
use tracing::info;

use athenas_core::{AthenasError, HardwareInfo, Result};

use crate::backend::{Backend, ModelInfo};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, CompletionRequest, CompletionResponse,
    InferenceStats, ModelLoadConfig, StreamChunk,
};

/// vLLM backend — manages a vLLM Python subprocess for high-throughput inference
pub struct VllmBackend {
    hardware: HardwareInfo,
    loaded: bool,
    model_path: String,
    model_name: String,
    context_size: u32,
    server_handle: Option<tokio::process::Child>,
    server_port: u16,
    client: reqwest::Client,
}

impl VllmBackend {
    pub fn new(hardware: &HardwareInfo) -> Self {
        Self {
            hardware: hardware.clone(),
            loaded: false,
            model_path: String::new(),
            model_name: String::new(),
            context_size: 4096,
            server_handle: None,
            server_port: 0,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap(),
        }
    }

    fn find_vllm(&self) -> Option<String> {
        for cmd in &["vllm", "python3", "python"] {
            if which::which(cmd).is_ok() {
                return Some(cmd.to_string());
            }
        }
        None
    }

    async fn start_server(&mut self, config: &ModelLoadConfig) -> Result<()> {
        let python_bin = self.find_vllm().ok_or_else(|| {
            AthenasError::Backend(
                "Python/vLLM not found. Install vLLM: pip install vllm".to_string(),
            )
        })?;

        self.server_port = find_free_port();

        let mut cmd = tokio::process::Command::new(&python_bin);
        cmd.arg("-m").arg("vllm.entrypoints.openai.api_server")
            .arg("--model").arg(&config.model_path)
            .arg("--port").arg(self.server_port.to_string())
            .arg("--host").arg("127.0.0.1")
            .arg("--max-model-len").arg(config.context_size.to_string());

        // GPU configuration
        if self.hardware.has_cuda {
            info!("Configuring vLLM with CUDA support");
        } else if self.hardware.has_rocm {
            info!("Configuring vLLM with ROCm support");
        }

        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        info!("Starting vLLM server on port {} with model: {}", self.server_port, config.model_path);

        let child = cmd.spawn().map_err(|e| {
            AthenasError::Backend(format!("Failed to start vLLM: {}", e))
        })?;

        self.server_handle = Some(child);

        // Wait for vLLM to be ready (can take longer than llama.cpp)
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        for _attempt in 0..60 {
            let url = format!("http://127.0.0.1:{}/health", self.server_port);
            if let Ok(resp) = self.client.get(&url).send().await {
                if resp.status().is_success() {
                    info!("vLLM server is ready on port {}", self.server_port);
                    return Ok(());
                }
            }
            if let Some(ref mut child) = self.server_handle {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        return Err(AthenasError::Backend(format!(
                            "vLLM exited early with status: {}. Make sure vLLM is installed: pip install vllm",
                            status
                        )));
                    }
                    Ok(None) => {}
                    Err(_) => {}
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        Err(AthenasError::Backend(
            "vLLM failed to start within 60 seconds".to_string(),
        ))
    }

    async fn stop_server(&mut self) -> Result<()> {
        if let Some(mut child) = self.server_handle.take() {
            child.kill().await.map_err(|e| {
                AthenasError::Backend(format!("Failed to kill vLLM: {}", e))
            })?;
            info!("vLLM server stopped");
        }
        Ok(())
    }

    fn server_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.server_port)
    }
}

#[async_trait]
impl Backend for VllmBackend {
    fn name(&self) -> &str {
        "vllm"
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

        let messages: Vec<serde_json::Value> = request.messages.iter().map(|m| {
            serde_json::json!({
                "role": m.role.to_string(),
                "content": m.content,
            })
        }).collect();

        let body = serde_json::json!({
            "model": self.model_path,
            "messages": messages,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "stream": false,
        });

        let url = format!("{}/v1/chat/completions", self.server_url());
        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| AthenasError::Backend(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("vLLM returned {}: {}", status, text)));
        }

        let result: serde_json::Value = resp.json().await
            .map_err(|e| AthenasError::Backend(format!("Invalid response: {}", e)))?;

        let content = result.pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = result.get("usage");
        let prompt_tokens = usage.and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let completion_tokens = usage.and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        Ok(ChatResponse {
            model: self.model_name.clone(),
            message: ChatMessage::assistant(content),
            stats: InferenceStats {
                tokens_generated: completion_tokens,
                tokens_prompt: prompt_tokens,
                time_total_ms: 0,
                tokens_per_second: 0.0,
            },
        })
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        let messages: Vec<serde_json::Value> = request.messages.iter().map(|m| {
            serde_json::json!({
                "role": m.role.to_string(),
                "content": m.content,
            })
        }).collect();

        let body = serde_json::json!({
            "model": self.model_path,
            "messages": messages,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "stream": true,
            "stream_options": {"include_usage": true},
        });

        let url = format!("{}/v1/chat/completions", self.server_url());
        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| AthenasError::Backend(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("vLLM returned {}: {}", status, text)));
        }

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                AthenasError::Backend(format!("Stream error: {}", e))
            })?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    let _ = tx.send(StreamChunk {
                        text: String::new(),
                        done: true,
                        stats: None,
                    }).await;
                    return Ok(());
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let content = json.pointer("/choices/0/delta/content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if !content.is_empty() {
                        let _ = tx.send(StreamChunk {
                            text: content.to_string(),
                            done: false,
                            stats: None,
                        }).await;
                    }

                    let finish = json.pointer("/choices/0/finish_reason")
                        .and_then(|v| v.as_str());
                    if let Some(reason) = finish {
                        if !reason.is_null_value() && reason != "null" {
                            let usage = json.get("usage");
                            let stats = usage.map(|u| InferenceStats {
                                tokens_generated: u.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                tokens_prompt: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                time_total_ms: 0,
                                tokens_per_second: 0.0,
                            });
                            let _ = tx.send(StreamChunk {
                                text: String::new(),
                                done: true,
                                stats,
                            }).await;
                            return Ok(());
                        }
                    }
                }
            }
        }

        let _ = tx.send(StreamChunk {
            text: String::new(),
            done: true,
            stats: None,
        }).await;
        Ok(())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        let body = serde_json::json!({
            "model": self.model_path,
            "prompt": request.prompt,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "stream": false,
        });

        let url = format!("{}/v1/completions", self.server_url());
        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| AthenasError::Backend(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("vLLM returned {}: {}", status, text)));
        }

        let result: serde_json::Value = resp.json().await
            .map_err(|e| AthenasError::Backend(format!("Invalid response: {}", e)))?;

        let text = result.pointer("/choices/0/text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = result.get("usage");
        let prompt_tokens = usage.and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let completion_tokens = usage.and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        Ok(CompletionResponse {
            model: self.model_name.clone(),
            text,
            stats: InferenceStats {
                tokens_generated: completion_tokens,
                tokens_prompt: prompt_tokens,
                time_total_ms: 0,
                tokens_per_second: 0.0,
            },
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        if !self.loaded {
            return Err(AthenasError::Backend("No model loaded".to_string()));
        }

        let body = serde_json::json!({
            "model": self.model_path,
            "prompt": request.prompt,
            "temperature": request.temperature.unwrap_or(0.7),
            "top_p": request.top_p.unwrap_or(0.9),
            "max_tokens": request.max_tokens.unwrap_or(2048),
            "stream": true,
        });

        let url = format!("{}/v1/completions", self.server_url());
        let resp = self.client.post(&url).json(&body).send().await
            .map_err(|e| AthenasError::Backend(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("vLLM returned {}: {}", status, text)));
        }

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                AthenasError::Backend(format!("Stream error: {}", e))
            })?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    let _ = tx.send(StreamChunk {
                        text: String::new(),
                        done: true,
                        stats: None,
                    }).await;
                    return Ok(());
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    let text = json.pointer("/choices/0/text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if !text.is_empty() {
                        let _ = tx.send(StreamChunk {
                            text: text.to_string(),
                            done: false,
                            stats: None,
                        }).await;
                    }

                    let finish = json.pointer("/choices/0/finish_reason")
                        .and_then(|v| v.as_str());
                    if let Some(reason) = finish {
                        if !reason.is_null_value() && reason != "null" {
                            let _ = tx.send(StreamChunk {
                                text: String::new(),
                                done: true,
                                stats: None,
                            }).await;
                            return Ok(());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn model_info(&self) -> Option<ModelInfo> {
        if self.loaded {
            Some(ModelInfo {
                name: self.model_name.clone(),
                context_size: self.context_size,
                gpu_layers: -1,
                backend_name: "vllm".to_string(),
            })
        } else {
            None
        }
    }
}

impl Drop for VllmBackend {
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
        .unwrap_or(9091)
}

trait IsNullValue {
    fn is_null_value(&self) -> bool;
}

impl IsNullValue for &str {
    fn is_null_value(&self) -> bool {
        self == &"null" || self.is_empty()
    }
}
