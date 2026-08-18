pub mod backend;
pub mod backend_setup;
pub mod llama_cpp;
pub mod remote;
pub mod types;
pub mod vllm;
pub mod whisper;

pub use backend::{Backend, BackendFactory, ModelInfo};
pub use remote::RemoteBackend;
pub use types::{
    ChatMessage, ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, ContentPart,
    EmbeddingData, EmbeddingInput, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, ImageUrl,
    InferenceConfig, InferenceStats, MessageContent, ModelLoadConfig, Role, StreamChunk,
    TokenizeRequest, TokenizeResponse, Tool, ToolCall, ToolCallFunction, ToolChoice,
    ToolChoiceFunction, ToolChoiceSpecific, ToolFunction, TranscriptionRequest,
    TranscriptionResponse, TranscriptionSegment,
};
pub use whisper::WhisperBackend;
