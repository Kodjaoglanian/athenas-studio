use async_trait::async_trait;
use tokio::sync::mpsc;

use athenas_core::Result;

use crate::types::{
    ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, ModelLoadConfig, StreamChunk,
};

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn is_loaded(&self) -> bool;

    async fn load_model(&mut self, config: ModelLoadConfig) -> Result<()>;
    async fn unload_model(&mut self) -> Result<()>;

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;
    async fn chat_stream(
        &self,
        request: ChatRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()>;

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;
    async fn complete_stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamChunk>,
    ) -> Result<()>;

    fn model_info(&self) -> Option<ModelInfo>;
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
            athenas_core::BackendType::Auto => {
                if hardware.has_cuda || hardware.has_rocm {
                    Ok(Box::new(crate::llama_cpp::LlamaCppBackend::new(hardware)))
                } else {
                    Ok(Box::new(crate::llama_cpp::LlamaCppBackend::new(hardware)))
                }
            }
        }
    }
}
