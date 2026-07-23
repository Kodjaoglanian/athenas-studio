pub mod backend;
pub mod backend_setup;
pub mod llama_cpp;
pub mod types;
pub mod vllm;

pub use backend::{Backend, BackendFactory, ModelInfo};
pub use types::{
    ChatMessage, ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, ContentPart,
    EmbeddingData, EmbeddingInput, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, ImageUrl,
    InferenceConfig, InferenceStats, MessageContent, ModelLoadConfig, Role, StreamChunk, Tool,
    ToolCall, ToolCallFunction, ToolChoice, ToolChoiceFunction, ToolChoiceSpecific, ToolFunction,
};
