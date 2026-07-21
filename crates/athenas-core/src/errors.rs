use thiserror::Error;

pub type Result<T> = std::result::Result<T, AthenasError>;

#[derive(Error, Debug)]
pub enum AthenasError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already exists: {0}")]
    ModelExists(String),

    #[error("Download error: {0}")]
    Download(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Hardware error: {0}")]
    Hardware(String),

    #[error("Server error: {0}")]
    Server(String),

    #[error("TUI error: {0}")]
    Tui(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("HuggingFace API error: {0}")]
    HfApi(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for AthenasError {
    fn from(e: serde_json::Error) -> Self {
        AthenasError::Serialization(e.to_string())
    }
}

impl From<toml::de::Error> for AthenasError {
    fn from(e: toml::de::Error) -> Self {
        AthenasError::Config(e.to_string())
    }
}

impl From<toml::ser::Error> for AthenasError {
    fn from(e: toml::ser::Error) -> Self {
        AthenasError::Config(e.to_string())
    }
}

impl From<rusqlite::Error> for AthenasError {
    fn from(e: rusqlite::Error) -> Self {
        AthenasError::Storage(e.to_string())
    }
}
