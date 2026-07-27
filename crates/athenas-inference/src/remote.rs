use async_trait::async_trait;
use tokio::sync::mpsc;

use athenas_core::{AthenasError, Result};

use crate::backend::{Backend, ModelInfo};
use crate::types::{
    ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, EmbeddingRequest,
    EmbeddingResponse, ModelLoadConfig, StreamChunk,
};

pub struct RemoteBackend {
    base_url: String,
    api_key: Option<String>,
    model_name: String,
    client: reqwest::Client,
}

impl RemoteBackend {
    pub fn new(host: &str, port: u16, model: &str, api_key: Option<&str>) -> Self {
        Self {
            base_url: format!("http://{}:{}", host, port),
            api_key: api_key.filter(|k| !k.is_empty()).map(|k| k.to_string()),
            model_name: model.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.api_key {
            req.bearer_auth(key)
        } else {
            req
        }
    }

    fn set_model(&self, mut body: serde_json::Value) -> serde_json::Value {
        if body.get("model").is_none() || body["model"].as_str() == Some("") {
            body["model"] = serde_json::Value::String(self.model_name.clone());
        }
        body
    }
}

#[async_trait]
impl Backend for RemoteBackend {
    fn name(&self) -> &str {
        "remote"
    }
    fn is_loaded(&self) -> bool {
        true
    }
    async fn load_model(&mut self, _: ModelLoadConfig) -> Result<()> {
        Ok(())
    }
    async fn unload_model(&mut self) -> Result<()> {
        Ok(())
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let mut body =
            serde_json::to_value(&request).map_err(|e| AthenasError::Backend(e.to_string()))?;
        body = self.set_model(body);
        body["stream"] = serde_json::Value::Bool(false);

        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .auth(self.client.post(&url).json(&body))
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("HTTP: {e}")))?;

        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("Server {s}: {t}")));
        }

        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AthenasError::Backend(e.to_string()))?;

        let content = v
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        let reasoning = v
            .pointer("/choices/0/message/reasoning_content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let finish = v
            .pointer("/choices/0/finish_reason")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string());

        let stats = crate::types::InferenceStats {
            tokens_generated: v
                .pointer("/usage/completion_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32,
            tokens_prompt: v
                .pointer("/usage/prompt_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32,
            time_total_ms: 0,
            tokens_per_second: 0.0,
        };

        let text = if !reasoning.is_empty() {
            format!("<think>{reasoning}</think>\n{content}")
        } else {
            content
        };
        Ok(ChatResponse {
            model: self.model_name.clone(),
            message: crate::types::ChatMessage {
                role: crate::types::Role::Assistant,
                content: crate::types::MessageContent::Text(text),
            },
            stats,
            tool_calls: None,
            finish_reason: finish,
        })
    }

    async fn chat_stream(&self, request: ChatRequest, tx: mpsc::Sender<StreamChunk>) -> Result<()> {
        let mut body =
            serde_json::to_value(&request).map_err(|e| AthenasError::Backend(e.to_string()))?;
        body = self.set_model(body);
        body["stream"] = serde_json::Value::Bool(true);

        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self
            .auth(self.client.post(&url).json(&body))
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("HTTP: {e}")))?;

        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("Server {s}: {t}")));
        }

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AthenasError::Backend(format!("Stream: {e}")))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf = buf[pos + 1..].to_string();
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        let _ = tx
                            .send(StreamChunk {
                                text: String::new(),
                                done: true,
                                is_reasoning: false,
                                stats: None,
                            })
                            .await;
                        return Ok(());
                    }
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        let delta = json.pointer("/choices/0/delta");
                        if let Some(r) = delta
                            .and_then(|d| d.get("reasoning_content"))
                            .and_then(|c| c.as_str())
                        {
                            if !r.is_empty() {
                                let _ = tx
                                    .send(StreamChunk {
                                        text: r.to_string(),
                                        done: false,
                                        is_reasoning: true,
                                        stats: None,
                                    })
                                    .await;
                            }
                        }
                        if let Some(c) = delta
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            if !c.is_empty() {
                                let _ = tx
                                    .send(StreamChunk {
                                        text: c.to_string(),
                                        done: false,
                                        is_reasoning: false,
                                        stats: None,
                                    })
                                    .await;
                            }
                        }
                        if json
                            .pointer("/choices/0/finish_reason")
                            .and_then(|f| f.as_str())
                            .is_some()
                        {
                            let stats = json.get("usage").map(|u| crate::types::InferenceStats {
                                tokens_generated: u
                                    .get("completion_tokens")
                                    .and_then(|t| t.as_u64())
                                    .unwrap_or(0)
                                    as u32,
                                tokens_prompt: u
                                    .get("prompt_tokens")
                                    .and_then(|t| t.as_u64())
                                    .unwrap_or(0)
                                    as u32,
                                time_total_ms: 0,
                                tokens_per_second: 0.0,
                            });
                            let _ = tx
                                .send(StreamChunk {
                                    text: String::new(),
                                    done: true,
                                    is_reasoning: false,
                                    stats,
                                })
                                .await;
                            return Ok(());
                        }
                    }
                }
            }
        }
        let _ = tx
            .send(StreamChunk {
                text: String::new(),
                done: true,
                is_reasoning: false,
                stats: None,
            })
            .await;
        Ok(())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut body =
            serde_json::to_value(&request).map_err(|e| AthenasError::Backend(e.to_string()))?;
        body = self.set_model(body);
        body["stream"] = serde_json::Value::Bool(false);

        let url = format!("{}/v1/completions", self.base_url);
        let resp = self
            .auth(self.client.post(&url).json(&body))
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("HTTP: {e}")))?;

        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("Server {s}: {t}")));
        }

        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AthenasError::Backend(e.to_string()))?;

        let text = v
            .pointer("/choices/0/text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let stats = crate::types::InferenceStats {
            tokens_generated: v
                .pointer("/usage/completion_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32,
            tokens_prompt: v
                .pointer("/usage/prompt_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(0) as u32,
            time_total_ms: 0,
            tokens_per_second: 0.0,
        };
        Ok(CompletionResponse {
            model: self.model_name.clone(),
            text,
            stats,
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()> {
        let mut body =
            serde_json::to_value(&request).map_err(|e| AthenasError::Backend(e.to_string()))?;
        body = self.set_model(body);
        body["stream"] = serde_json::Value::Bool(true);

        let url = format!("{}/v1/completions", self.base_url);
        let resp = self
            .auth(self.client.post(&url).json(&body))
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("HTTP: {e}")))?;

        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("Server {s}: {t}")));
        }

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| AthenasError::Backend(format!("Stream: {e}")))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf = buf[pos + 1..].to_string();
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        let _ = tx
                            .send(StreamChunk {
                                text: String::new(),
                                done: true,
                                is_reasoning: false,
                                stats: None,
                            })
                            .await;
                        return Ok(());
                    }
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(c) = json.pointer("/choices/0/text").and_then(|t| t.as_str()) {
                            if !c.is_empty() {
                                let _ = tx
                                    .send(StreamChunk {
                                        text: c.to_string(),
                                        done: false,
                                        is_reasoning: false,
                                        stats: None,
                                    })
                                    .await;
                            }
                        }
                        if json
                            .pointer("/choices/0/finish_reason")
                            .and_then(|f| f.as_str())
                            .is_some()
                        {
                            let stats = json.get("usage").map(|u| crate::types::InferenceStats {
                                tokens_generated: u
                                    .get("completion_tokens")
                                    .and_then(|t| t.as_u64())
                                    .unwrap_or(0)
                                    as u32,
                                tokens_prompt: u
                                    .get("prompt_tokens")
                                    .and_then(|t| t.as_u64())
                                    .unwrap_or(0)
                                    as u32,
                                time_total_ms: 0,
                                tokens_per_second: 0.0,
                            });
                            let _ = tx
                                .send(StreamChunk {
                                    text: String::new(),
                                    done: true,
                                    is_reasoning: false,
                                    stats,
                                })
                                .await;
                            return Ok(());
                        }
                    }
                }
            }
        }
        let _ = tx
            .send(StreamChunk {
                text: String::new(),
                done: true,
                is_reasoning: false,
                stats: None,
            })
            .await;
        Ok(())
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body =
            serde_json::to_value(&request).map_err(|e| AthenasError::Backend(e.to_string()))?;
        let resp = self
            .auth(self.client.post(&url).json(&body))
            .send()
            .await
            .map_err(|e| AthenasError::Backend(format!("HTTP: {e}")))?;

        if !resp.status().is_success() {
            let s = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(AthenasError::Backend(format!("Server {s}: {t}")));
        }

        resp.json()
            .await
            .map_err(|e| AthenasError::Backend(format!("Parse: {e}")))
    }

    fn model_info(&self) -> Option<ModelInfo> {
        Some(ModelInfo {
            name: self.model_name.clone(),
            context_size: 0,
            gpu_layers: -1,
            backend_name: "remote".to_string(),
        })
    }

    fn boxed_clone(&self) -> Box<dyn Backend> {
        Box::new(RemoteBackend {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            model_name: self.model_name.clone(),
            client: self.client.clone(),
        })
    }
}
