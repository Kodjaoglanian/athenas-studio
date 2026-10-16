pub mod backend;
pub mod backend_setup;
pub mod llama_cpp;
// No ORT binaries for x86_64 macOS or musl; Windows ARM64 is omitted too
// (BackendFactory errors on Onnx requests on those targets).
#[cfg(not(any(
    all(target_os = "macos", target_arch = "x86_64"),
    target_env = "musl",
    all(target_os = "windows", target_arch = "aarch64"),
)))]
pub mod onnx;
pub mod remote;
pub mod types;
pub mod vllm;
pub mod whisper;

pub use backend::{Backend, BackendFactory, ModelInfo};
#[cfg(not(any(
    all(target_os = "macos", target_arch = "x86_64"),
    target_env = "musl",
    all(target_os = "windows", target_arch = "aarch64"),
)))]
pub use onnx::GPU_OFFLOAD_CAPABLE;
/// Whether this build can run ONNX models on a GPU. Always false where
/// the onnx backend isn't compiled (x86_64 macOS, musl, Windows ARM64).
#[cfg(any(
    all(target_os = "macos", target_arch = "x86_64"),
    target_env = "musl",
    all(target_os = "windows", target_arch = "aarch64"),
))]
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
