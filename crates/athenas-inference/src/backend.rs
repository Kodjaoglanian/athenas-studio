use async_trait::async_trait;
use tokio::sync::mpsc;

use athenas_core::Result;

use crate::types::{
    ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, EmbeddingRequest,
    EmbeddingResponse, ModelLoadConfig, StreamChunk, TokenizeRequest, TokenizeResponse,
};

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn is_loaded(&self) -> bool;

    async fn load_model(&mut self, config: ModelLoadConfig) -> Result<()>;
    async fn unload_model(&mut self) -> Result<()>;

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;
    async fn chat_stream(&self, request: ChatRequest, tx: mpsc::Sender<StreamChunk>) -> Result<()>;

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    async fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()>;

    /// Generate embeddings for the given input text(s).
    /// Default implementation returns an error — backends override if supported.
    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        Err(athenas_core::AthenasError::Backend(
            "Embeddings not supported by this backend".to_string(),
        ))
    }

    /// Tokenize text into token IDs using the loaded model's tokenizer.
    /// Default implementation returns an error — backends override if supported.
    async fn tokenize(&self, _request: TokenizeRequest) -> Result<TokenizeResponse> {
        Err(athenas_core::AthenasError::Backend(
            "Tokenization not supported by this backend".to_string(),
        ))
    }

    fn model_info(&self) -> Option<ModelInfo>;

    /// Check if the backend's inference server is alive and responsive.
    /// Default implementation returns true (assumes healthy).
    /// Backends with a subprocess (like llama-server) should override
    /// this to do a quick health check.
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }

    /// Clone into a boxed trait object (for spawning background tasks).
    /// Only needs to support read-only operations (chat, complete, stream).
    fn boxed_clone(&self) -> Box<dyn Backend>;
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub context_size: u32,
    pub gpu_layers: i32,
    pub backend_name: String,
}

pub struct BackendFactory;

impl BackendFactory {
    pub fn create(
        backend_type: athenas_core::BackendType,
        hardware: &athenas_core::HardwareInfo,
    ) -> Result<Box<dyn Backend>> {
        match backend_type {
            athenas_core::BackendType::LlamaCpp => {
                Ok(Box::new(crate::llama_cpp::LlamaCppBackend::new(hardware)))
            }
            athenas_core::BackendType::Vllm => {
                Ok(Box::new(crate::vllm::VllmBackend::new(hardware)))
            }
            athenas_core::BackendType::Onnx => {
                Ok(Box::new(crate::onnx::OnnxBackend::new(hardware)))
            }
            athenas_core::BackendType::Auto => {
                Ok(Box::new(crate::llama_cpp::LlamaCppBackend::new(hardware)))
            }
        }
    }

    /// Like `create`, but resolves `BackendType::Auto` from the model path:
    /// `.onnx` files and ONNX model directories get the ONNX backend,
    /// everything else falls back to llama.cpp. An explicit backend type
    /// is honored as-is.
    pub fn create_for_model(
        backend_type: athenas_core::BackendType,
        hardware: &athenas_core::HardwareInfo,
        model_path: &str,
    ) -> Result<Box<dyn Backend>> {
        let resolved = match backend_type {
            athenas_core::BackendType::Auto
                if athenas_core::is_onnx_model_path(std::path::Path::new(model_path)) =>
            {
                athenas_core::BackendType::Onnx
            }
            other => other,
        };
        Self::create(resolved, hardware)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use athenas_core::{BackendType, HardwareInfo};

    fn hw() -> HardwareInfo {
        HardwareInfo {
            cpus: 4,
            memory_total_mb: 16 * 1024,
            memory_available_mb: 16 * 1024,
            gpus: Vec::new(),
            has_cuda: false,
            has_rocm: false,
            has_vulkan: false,
            has_metal: false,
        }
    }

    #[test]
    fn auto_resolves_onnx_file_to_onnx() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("model.onnx");
        std::fs::write(&f, b"x").unwrap();
        let b = BackendFactory::create_for_model(BackendType::Auto, &hw(), f.to_str().unwrap())
            .unwrap();
        assert_eq!(b.name(), "onnx");
    }

    #[test]
    fn auto_resolves_onnx_dir_to_onnx() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("genai_config.json"), b"{}").unwrap();
        let b = BackendFactory::create_for_model(
            BackendType::Auto,
            &hw(),
            dir.path().to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(b.name(), "onnx");
    }

    #[test]
    fn auto_falls_back_to_llamacpp() {
        let b =
            BackendFactory::create_for_model(BackendType::Auto, &hw(), "/nonexistent/model.gguf")
                .unwrap();
        assert_eq!(b.name(), "llama.cpp");
    }

    #[test]
    fn explicit_backend_is_honored() {
        let b = BackendFactory::create_for_model(
            BackendType::LlamaCpp,
            &hw(),
            "/nonexistent/model.onnx",
        )
        .unwrap();
        assert_eq!(b.name(), "llama.cpp");
    }
}
