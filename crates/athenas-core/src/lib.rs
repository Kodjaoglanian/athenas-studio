pub mod config;
pub mod errors;
pub mod hardware;
pub mod model_registry;

pub use config::{
    AppConfig, BackendType, GpuRuntime, InferenceConfig, OtelConfig, SemanticCacheConfig,
    ServerConfig, VectorStoreServerConfig,
};
pub use errors::{AthenasError, Result};
pub use hardware::{
    detect_memory_mb, estimate_model_memory, estimate_model_ram_mb, is_apu_name, GpuInfo,
    HardwareDetector, HardwareInfo, ModelMemoryEstimate,
};
pub use model_registry::{dir_size, is_onnx_model_path, ModelFormat, ModelInfo, ModelRegistry};
