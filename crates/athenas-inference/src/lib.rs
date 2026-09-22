pub mod backend;
pub mod backend_setup;
pub mod llama_cpp;
// No ORT binaries exist for x86_64 macOS or musl targets — the onnx
// backend is compiled out there (BackendFactory errors on Onnx requests).
#[cfg(not(any(all(target_os = "macos", target_arch = "x86_64"), target_env = "musl")))]
pub mod onnx;
pub mod remote;
pub mod types;
pub mod vllm;
pub mod whisper;

pub use backend::{Backend, BackendFactory, ModelInfo};
#[cfg(not(any(all(target_os = "macos", target_arch = "x86_64"), target_env = "musl")))]
pub use onnx::GPU_OFFLOAD_CAPABLE;
/// Whether this build can run ONNX models on a GPU. Always false where
/// the onnx backend isn't compiled (x86_64 macOS, musl).
#[cfg(any(all(target_os = "macos", target_arch = "x86_64"), target_env = "musl"))]
pub const GPU_OFFLOAD_CAPABLE: bool = false;
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
