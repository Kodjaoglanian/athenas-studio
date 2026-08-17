pub mod config;
pub mod errors;
pub mod hardware;
pub mod model_registry;
pub mod storage;

pub use config::{
    AppConfig, BackendType, GpuRuntime, InferenceConfig, OtelConfig, ServerConfig,
    VectorStoreServerConfig,
};
pub use errors::{AthenasError, Result};
pub use hardware::{
    detect_memory_mb, estimate_model_ram_mb, is_apu_name, GpuInfo, HardwareDetector, HardwareInfo,
};
pub use model_registry::{ModelFormat, ModelInfo, ModelRegistry};
pub use storage::Database;
